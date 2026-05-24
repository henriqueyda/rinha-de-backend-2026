package main

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/bytedance/sonic"
	"github.com/valyala/fasthttp"
)

const (
	Dimensions = 14

	KClusters = 8
	NProbe    = 1
	KNN       = 5
	MaxIters  = 5
)

var (
	index   *IVFIndex
	ready   atomic.Bool
	norm    Normalization
	mccRisk map[string]float32
)

type Reference struct {
	Vector [Dimensions]float32 `json:"vector"`
	Label  string              `json:"label"`
}

type Vector struct {
	Values [Dimensions]float32
	Fraud  bool
}

type Cluster struct {
	Centroid [Dimensions]float32
	Vectors  []Vector
}

type IVFIndex struct {
	Clusters []Cluster
}

type TransactionRequest struct {
	ID string `json:"id"`

	Transaction struct {
		Amount       float32 `json:"amount"`
		Installments int     `json:"installments"`
		RequestedAt  string  `json:"requested_at"`
	} `json:"transaction"`

	Customer struct {
		AvgAmount      float32  `json:"avg_amount"`
		TxCount24h     int      `json:"tx_count_24h"`
		KnownMerchants []string `json:"known_merchants"`
	} `json:"customer"`

	Merchant struct {
		ID        string  `json:"id"`
		MCC       string  `json:"mcc"`
		AvgAmount float32 `json:"avg_amount"`
	} `json:"merchant"`

	Terminal struct {
		IsOnline    bool    `json:"is_online"`
		CardPresent bool    `json:"card_present"`
		KmFromHome  float32 `json:"km_from_home"`
	} `json:"terminal"`

	LastTransaction *struct {
		RequestedAt   string  `json:"requested_at"`
		KmFromCurrent float32 `json:"km_from_current"`
	} `json:"last_transaction"`
}

type Normalization struct {
	MaxAmount            float32 `json:"max_amount"`
	MaxInstallments      float32 `json:"max_installments"`
	AmountVsAvgRatio     float32 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float32 `json:"max_minutes"`
	MaxKm                float32 `json:"max_km"`
	MaxTxCount24h        float32 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float32 `json:"max_merchant_avg_amount"`
}

type Neighbor struct {
	Dist  float32
	Fraud bool
}

type MaxHeap []Neighbor

func (h MaxHeap) Len() int {
	return len(h)
}

func (h MaxHeap) Less(i, j int) bool {
	return h[i].Dist > h[j].Dist
}

func (h MaxHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *MaxHeap) Push(x interface{}) {
	*h = append(*h, x.(Neighbor))
}

func (h *MaxHeap) Pop() interface{} {
	old := *h

	n := len(old)

	item := old[n-1]

	*h = old[:n-1]

	return item
}

func main() {
	// Dispara a inicialização pesada em background exatamente como você fazia
	go initialize()

	// MANTIDO: O pprof original continua escutando na porta 6060 usando o net/http padrão
	go func() {
		log.Println(http.ListenAndServe("0.0.0.0:6060", nil))
	}()

	// 3. O Roteador central do Fasthttp avalia o caminho da URL por bytes
	requestHandler := func(ctx *fasthttp.RequestCtx) {
		switch string(ctx.Path()) {
		case "/health":
			healthHandler(ctx)
		case "/fraud-score":
			fraudScoreHandler(ctx)
		default:
			ctx.SetStatusCode(fasthttp.StatusNotFound)
			ctx.SetBodyString(`{"error": "not found"}`)
		}
	}

	fmt.Println("Server listening on :8080")

	// 4. Sobe o servidor de alta performance na porta de produção
	if err := fasthttp.ListenAndServe(":8080", requestHandler); err != nil {
		log.Fatalf("Erro ao iniciar o servidor fasthttp: %s", err)
	}
}

// func main() {
// 	go initialize()

// 	go func() {
// 		log.Println(http.ListenAndServe("0.0.0.0:6060", nil))
// 	}()

// 	http.HandleFunc("/health", healthHandler)
// 	http.HandleFunc("/fraud-score", fraudScoreHandler)

// 	fmt.Println("Server listening on :8080")

// 	log.Fatal(http.ListenAndServe(":8080", nil))
// }

func initialize() {
	fmt.Println("Loading normalization...")

	var err error

	norm, err = loadNormalization("resources/normalization.json")
	if err != nil {
		panic(err)
	}

	fmt.Println("Loading MCC risk...")

	mccRisk, err = loadMccRisk("resources/mcc_risk.json")
	if err != nil {
		panic(err)
	}

	fmt.Println("Loading references dataset...")

	// vectors, err := loadDataset("resources/references.json.gz")
	// if err != nil {
	// 	panic(err)
	// }

	vectors, err := loadExampleReferences("resources/example-references.json")
	if err != nil {
		panic(err)
	}

	fmt.Printf("Loaded %d vectors\n", len(vectors))

	fmt.Println("Building IVF index...")

	index = buildIVF(vectors, KClusters)

	fmt.Println("IVF ready")

	ready.Store(true)
}

// func healthHandler(w http.ResponseWriter, r *http.Request) {
// 	if !ready.Load() {
// 		w.WriteHeader(http.StatusServiceUnavailable)

// 		json.NewEncoder(w).Encode(map[string]string{
// 			"status": "loading",
// 		})

// 		return
// 	}

// 	w.WriteHeader(http.StatusOK)

// 	json.NewEncoder(w).Encode(map[string]string{
// 		"status": "ready",
// 	})
// }

func healthHandler(ctx *fasthttp.RequestCtx) {
	ctx.SetContentType("application/json")

	if !ready.Load() {
		ctx.SetStatusCode(fasthttp.StatusServiceUnavailable)
		// Alocação Zero: escreve a string direta no buffer de resposta
		ctx.SetBodyString(`{"status": "loading"}`)
		return
	}

	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetBodyString(`{"status": "ready"}`)
}

var requestPool = sync.Pool{
	New: func() interface{} {
		return new(TransactionRequest)
	},
}

func fraudScoreHandler(ctx *fasthttp.RequestCtx) {
	// 1. Proteção caso batam na rota antes do IVF terminar de carregar em background
	if !ready.Load() {
		ctx.SetStatusCode(fasthttp.StatusServiceUnavailable)
		ctx.SetBodyString(`{"error": "index still loading"}`)
		return
	}

	// 2. Pega uma struct limpa do Pool
	req := requestPool.Get().(*TransactionRequest)

	// Garante a limpeza do slice e a devolução ao pool no final do fluxo
	defer func() {
		req.Customer.KnownMerchants = req.Customer.KnownMerchants[:0]
		requestPool.Put(req)
	}()

	// 3. Resgata os bytes do corpo sem alocações extras de I/O
	bodyBytes := ctx.PostBody()

	// 4. Parse ultra veloz com o Sonic
	if err := sonic.Unmarshal(bodyBytes, req); err != nil {
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		ctx.SetBodyString(`{"error": "invalid json format"}`)
		return
	}

	// 5. SUA LÓGICA DE NEGÓCIO DE VOLTA:
	// Desreferenciamos o ponteiro (*req) pois a sua função Vectorize espera a struct por valor
	vector := Vectorize(*req, norm, mccRisk)

	// Executa a busca vetorial otimizada na Stack
	approved, score := index.Search(vector)

	// 6. RESPOSTA DINÂMICA ULTRA OTMIZADA:
	// Em vez de criar um map[string]interface{} (que geraria alocações no heap),
	// nós escrevemos diretamente o JSON formatado no Writer de resposta do contexto.
	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentType("application/json")
	fmt.Fprintf(ctx, `{"approved":%t,"fraud_score":%.2f}`, approved, score)
}

// func fraudScoreHandler(w http.ResponseWriter, r *http.Request) {
// 	if !ready.Load() {
// 		http.Error(
// 			w,
// 			"index still loading",
// 			http.StatusServiceUnavailable,
// 		)

// 		return
// 	}

// 	req := requestPool.Get().(*TransactionRequest)

// 	defer func() {
// 		// Limpa os slices internos para não vazar dados entre requisições
// 		req.Customer.KnownMerchants = req.Customer.KnownMerchants[:0]
// 		requestPool.Put(req)
// 	}()

// 	bodyBytes, err := io.ReadAll(r.Body)
// 	if err != nil {
// 		http.Error(w, err.Error(), http.StatusBadRequest)
// 		return
// 	}

// 	// O Sonic faz o Unmarshal direto nos bytes sem alocar scanners internos
// 	err = sonic.Unmarshal(bodyBytes, req)

// 	vector := Vectorize(
// 		*req,
// 		norm,
// 		mccRisk,
// 	)

// 	approved, score := index.Search(vector)

// 	response := map[string]interface{}{
// 		"approved":    approved,
// 		"fraud_score": score,
// 	}

// 	w.Header().Set("Content-Type", "application/json")

// 	json.NewEncoder(w).Encode(response)
// }

func loadDataset(path string) ([]Vector, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	decoder := json.NewDecoder(gz)

	_, err = decoder.Token()
	if err != nil {
		return nil, err
	}

	vectors := make([]Vector, 0, 3_000_000)

	for decoder.More() {
		var ref Reference

		if err := decoder.Decode(&ref); err != nil {
			return nil, err
		}

		vectors = append(vectors, Vector{
			Values: ref.Vector,
			Fraud:  ref.Label == "fraud",
		})
	}

	return vectors, nil
}

func loadExampleReferences(path string) ([]Vector, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var refs []Reference

	if err := json.NewDecoder(file).Decode(&refs); err != nil {
		return nil, err
	}

	vectors := make([]Vector, 0, len(refs))

	for _, ref := range refs {
		vectors = append(vectors, Vector{
			Values: ref.Vector,
			Fraud:  ref.Label == "fraud",
		})
	}

	return vectors, nil
}

func loadNormalization(path string) (Normalization, error) {
	var norm Normalization

	file, err := os.Open(path)
	if err != nil {
		return norm, err
	}
	defer file.Close()

	err = json.NewDecoder(file).Decode(&norm)

	return norm, err
}

func loadMccRisk(path string) (map[string]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var risks map[string]float32

	err = json.NewDecoder(file).Decode(&risks)

	return risks, err
}

func buildIVF(vectors []Vector, k int) *IVFIndex {
	clusters := make([]Cluster, k)

	for i := 0; i < k; i++ {
		randomVec := vectors[rand.Intn(len(vectors))]
		clusters[i].Centroid = randomVec.Values
	}

	for iter := 0; iter < MaxIters; iter++ {
		fmt.Printf("KMeans iteration %d/%d\n", iter+1, MaxIters)

		for i := range clusters {
			clusters[i].Vectors = clusters[i].Vectors[:0]
		}

		for _, vec := range vectors {
			bestCluster := 0
			bestDist := float32(math.MaxFloat32)

			for c := range clusters {
				dist := squaredDistance(
					vec.Values,
					clusters[c].Centroid,
				)

				if dist < bestDist {
					bestDist = dist
					bestCluster = c
				}
			}

			clusters[bestCluster].Vectors = append(
				clusters[bestCluster].Vectors,
				vec,
			)
		}

		for c := range clusters {
			if len(clusters[c].Vectors) == 0 {
				continue
			}

			var newCentroid [Dimensions]float32

			for _, vec := range clusters[c].Vectors {
				for d := 0; d < Dimensions; d++ {
					newCentroid[d] += vec.Values[d]
				}
			}

			for d := 0; d < Dimensions; d++ {
				newCentroid[d] /= float32(len(clusters[c].Vectors))
			}

			clusters[c].Centroid = newCentroid
		}
	}

	return &IVFIndex{
		Clusters: clusters,
	}
}

// func (idx *IVFIndex) Search(query [Dimensions]float32) (bool, float32) {
// 	type ClusterDistance struct {
// 		Index int
// 		Dist  float32
// 	}

// 	clusterDists := make([]ClusterDistance, len(idx.Clusters))
// 	for i, cluster := range idx.Clusters {
// 		clusterDists[i] = ClusterDistance{
// 			Index: i,
// 			Dist: squaredDistance(
// 				query,
// 				cluster.Centroid,
// 			),
// 		}
// 	}
// 	sort.Slice(clusterDists, func(i, j int) bool {
// 		return clusterDists[i].Dist < clusterDists[j].Dist
// 	})

// 	topK := &MaxHeap{}

// 	heap.Init(topK)

// 	for i := 0; i < NProbe; i++ {
// 		cluster := idx.Clusters[clusterDists[i].Index]

// 		for _, vec := range cluster.Vectors {

// 			dist := squaredDistance(
// 				query,
// 				vec.Values,
// 			)

// 			neighbor := Neighbor{
// 				Dist:  dist,
// 				Fraud: vec.Fraud,
// 			}

// 			if topK.Len() < KNN {
// 				heap.Push(topK, neighbor)
// 				continue
// 			}

// 			if dist < (*topK)[0].Dist {
// 				heap.Pop(topK)
// 				heap.Push(topK, neighbor)
// 			}
// 		}
// 	}

// 	frauds := 0

// 	for _, n := range *topK {
// 		if n.Fraud {
// 			frauds++
// 		}
// 	}

// 	score := float32(frauds) / float32(KNN)

// 	return score < 0.6, score
// }

func (idx *IVFIndex) Search(query [Dimensions]float32) (bool, float32) {
	// 1. Como NProbe = 1, evitamos o 'make' e o 'sort.Slice'.
	// Encontramos o cluster mais próximo com um loop simples O(N).
	bestClusterIdx := 0
	bestClusterDist := float32(math.MaxFloat32)

	for i, cluster := range idx.Clusters {
		dist := squaredDistance(query, cluster.Centroid)
		if dist < bestClusterDist {
			bestClusterDist = dist
			bestClusterIdx = i
		}
	}

	cluster := idx.Clusters[bestClusterIdx]

	// 2. Criamos um array fixo na Stack para o TopK.
	// Isso elimina completamente o container/heap e suas alocações.
	var topK [KNN]Neighbor
	for i := range topK {
		topK[i].Dist = math.MaxFloat32
	}

	for _, vec := range cluster.Vectors {
		dist := squaredDistance(query, vec.Values)

		// Se a distância atual for maior ou igual ao pior vizinho que já temos, ignora
		if dist >= topK[KNN-1].Dist {
			continue
		}

		// Inserção ordenada direta no array (Insertion Sort manual para tamanho 5)
		for j := 0; j < KNN; j++ {
			if dist < topK[j].Dist {
				// Desloca os elementos restantes para a direita
				for k := KNN - 1; k > j; k-- {
					topK[k] = topK[k-1]
				}
				topK[j] = Neighbor{Dist: dist, Fraud: vec.Fraud}
				break
			}
		}
	}

	// 3. Contabiliza os resultados
	frauds := 0
	for i := 0; i < KNN; i++ {
		if topK[i].Dist == math.MaxFloat32 {
			continue // Evita slots vazios caso o cluster tenha menos que 5 elementos
		}
		if topK[i].Fraud {
			frauds++
		}
	}

	score := float32(frauds) / float32(KNN)
	return score < 0.6, score
}

func squaredDistance(
	a [Dimensions]float32,
	b [Dimensions]float32,
) float32 {

	var sum float32

	for i := 0; i < Dimensions; i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return sum
}

func Vectorize(
	req TransactionRequest,
	norm Normalization,
	mccRisk map[string]float32,
) [14]float32 {

	var vec [14]float32

	vec[0] = clamp(req.Transaction.Amount / norm.MaxAmount)

	vec[1] = clamp(
		float32(req.Transaction.Installments) /
			norm.MaxInstallments,
	)

	if req.Customer.AvgAmount > 0 {
		ratio := (req.Transaction.Amount / req.Customer.AvgAmount) / norm.AmountVsAvgRatio

		vec[2] = clamp(ratio)
	}

	t, _ := time.Parse(
		time.RFC3339,
		req.Transaction.RequestedAt,
	)

	vec[3] = float32(t.UTC().Hour()) / 23.0

	weekday := int(t.UTC().Weekday())
	weekday = (weekday + 6) % 7

	vec[4] = float32(weekday) / 6.0

	if req.LastTransaction == nil {
		vec[5] = -1
		vec[6] = -1
	} else {
		lastTime, _ := time.Parse(
			time.RFC3339,
			req.LastTransaction.RequestedAt,
		)

		minutes := float32(
			t.Sub(lastTime).Minutes(),
		)

		vec[5] = clamp(minutes / norm.MaxMinutes)

		vec[6] = clamp(
			req.LastTransaction.KmFromCurrent /
				norm.MaxKm,
		)
	}

	vec[7] = clamp(
		req.Terminal.KmFromHome / norm.MaxKm,
	)

	vec[8] = clamp(
		float32(req.Customer.TxCount24h) /
			norm.MaxTxCount24h,
	)

	if req.Terminal.IsOnline {
		vec[9] = 1
	}

	if req.Terminal.CardPresent {
		vec[10] = 1
	}

	known := false

	for _, merchant := range req.Customer.KnownMerchants {
		if merchant == req.Merchant.ID {
			known = true
			break
		}
	}

	if !known {
		vec[11] = 1
	}

	risk, ok := mccRisk[req.Merchant.MCC]

	if !ok {
		risk = 0.5
	}

	vec[12] = risk

	vec[13] = clamp(
		req.Merchant.AvgAmount /
			norm.MaxMerchantAvgAmount,
	)

	return vec
}

func clamp(v float32) float32 {
	if v < 0 {
		return 0
	}

	if v > 1 {
		return 1
	}

	return v
}
