package main

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net"
	"net/http"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

const (
	Dimensions = 14
	KClusters  = 1024
	NProbe     = 3
	KNN        = 5
	MaxIters   = 20
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
	Values [Dimensions]uint8
	Fraud  bool
}

type Cluster struct {
	Centroid [Dimensions]uint8
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
	Dist  int32
	Fraud bool
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "preprocess" {
		runPreprocess()
		return
	}
	runtime.GOMAXPROCS(1)
	go initialize()

	http.HandleFunc("/ready", readyHandler)
	http.HandleFunc("/fraud-score", fraudScoreHandler)

	socketPath := os.Getenv("API_SOCKET")
	if socketPath != "" {
		_ = os.Remove(socketPath)

		ln, err := net.Listen("unix", socketPath)
		if err != nil {
			log.Fatal(err)
		}
		defer ln.Close()

		if err := os.Chmod(socketPath, 0o666); err != nil {
			log.Fatal(err)
		}

		fmt.Println("Server listening on unix socket", socketPath)
		log.Fatal(http.Serve(ln, nil))
	}

	fmt.Println("Server listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func runPreprocess() {
	fmt.Println("Beginning preprocessing...")

	vectors, err := loadDataset("resources/references.json.gz")
	if err != nil {
		panic(err)
	}
	fmt.Printf("Loaded %d vectors\n", len(vectors))

	fmt.Println("Building IVF index (KMeans)...")
	idx := buildIVF(vectors, KClusters)

	fmt.Println("Saving index.bin...")
	file, err := os.Create("resources/index.bin")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	err = binary.Write(file, binary.LittleEndian, int32(len(idx.Clusters)))
	if err != nil {
		panic(err)
	}

	for _, c := range idx.Clusters {
		binary.Write(file, binary.LittleEndian, c.Centroid)
		binary.Write(file, binary.LittleEndian, int32(len(c.Vectors)))
		binary.Write(file, binary.LittleEndian, c.Vectors)
	}

	fmt.Println("Preprocessing finished!")
}

func loadIndexBin(path string) (*IVFIndex, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var idx IVFIndex
	var numClusters int32

	if err := binary.Read(file, binary.LittleEndian, &numClusters); err != nil {
		return nil, err
	}

	idx.Clusters = make([]Cluster, numClusters)
	for i := 0; i < int(numClusters); i++ {
		if err := binary.Read(file, binary.LittleEndian, &idx.Clusters[i].Centroid); err != nil {
			return nil, err
		}

		var numVectors int32
		if err := binary.Read(file, binary.LittleEndian, &numVectors); err != nil {
			return nil, err
		}

		idx.Clusters[i].Vectors = make([]Vector, numVectors)
		if err := binary.Read(file, binary.LittleEndian, &idx.Clusters[i].Vectors); err != nil {
			return nil, err
		}
	}

	return &idx, nil
}

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

	index, err = loadIndexBin("resources/index.bin")
	if err != nil {
		panic(fmt.Errorf("failed to read index.bin: %v", err))
	}
	ready.Store(true)
}

func readyHandler(w http.ResponseWriter, r *http.Request) {
	if !ready.Load() {
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}

	w.WriteHeader(http.StatusOK)
}

type FraudResponse struct {
	Approved   bool    `json:"approved"`
	FraudScore float32 `json:"fraud_score"`
}

var pool = sync.Pool{
	New: func() any {
		return new(TransactionRequest)
	},
}

func fraudScoreHandler(w http.ResponseWriter, r *http.Request) {
	if !ready.Load() {
		http.Error(
			w,
			"index still loading",
			http.StatusServiceUnavailable,
		)
		return
	}

	req := pool.Get().(*TransactionRequest)
	defer func() {
		req.Customer.KnownMerchants = req.Customer.KnownMerchants[:0]
		req.LastTransaction = nil
		pool.Put(req)
	}()
	if err := json.NewDecoder(r.Body).Decode(req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	vector := Vectorize(
		req,
		norm,
		mccRisk,
	)

	approved, score := index.Search(vector)

	resp := FraudResponse{
		Approved:   approved,
		FraudScore: score,
	}

	w.Header().Set("Content-Type", "application/json")

	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

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

		var vals [Dimensions]uint8
		for i, v := range ref.Vector {
			vals[i] = toUint8(v)
		}

		vectors = append(vectors, Vector{
			Values: vals,
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
			bestDist := int32(math.MaxInt32)

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

			var newCentroidSum [Dimensions]int

			for _, vec := range clusters[c].Vectors {
				for d := 0; d < Dimensions; d++ {
					newCentroidSum[d] += int(vec.Values[d])
				}
			}

			for d := 0; d < Dimensions; d++ {
				clusters[c].Centroid[d] = uint8(newCentroidSum[d] / len(clusters[c].Vectors))
			}
		}
	}

	return &IVFIndex{
		Clusters: clusters,
	}
}

func (idx *IVFIndex) Search(query [Dimensions]uint8) (bool, float32) {
	type LocalClusterDist struct {
		Index int
		Dist  int32
	}
	var clusterDists [KClusters]LocalClusterDist

	for i := range idx.Clusters {
		clusterDists[i] = LocalClusterDist{
			Index: i,
			Dist:  squaredDistance(query, idx.Clusters[i].Centroid),
		}
	}

	for i := 1; i < KClusters; i++ {
		key := clusterDists[i]
		j := i - 1
		for j >= 0 && clusterDists[j].Dist > key.Dist {
			clusterDists[j+1] = clusterDists[j]
			j--
		}
		clusterDists[j+1] = key
	}

	var topNeighbors [KNN]Neighbor
	for i := range topNeighbors {
		topNeighbors[i].Dist = math.MaxInt32
	}
	count := 0

	for p := 0; p < NProbe; p++ {
		clusterIdx := clusterDists[p].Index
		cluster := &idx.Clusters[clusterIdx]

		for i := range cluster.Vectors {
			vec := &cluster.Vectors[i]
			dist := squaredDistance(query, vec.Values)

			if dist < topNeighbors[KNN-1].Dist {
				pos := KNN - 1
				for pos > 0 && dist < topNeighbors[pos-1].Dist {
					pos--
				}
				for j := KNN - 1; j > pos; j-- {
					topNeighbors[j] = topNeighbors[j-1]
				}
				topNeighbors[pos] = Neighbor{Dist: dist, Fraud: vec.Fraud}
				if count < KNN {
					count++
				}
			}
		}
	}

	frauds := 0
	for i := 0; i < count; i++ {
		if topNeighbors[i].Fraud {
			frauds++
		}
	}

	score := float32(frauds) / float32(KNN)
	return score < 0.6, score
}

func squaredDistance(
	a [Dimensions]uint8,
	b [Dimensions]uint8,
) int32 {

	var sum int32

	for i := 0; i < Dimensions; i++ {
		diff := int32(a[i]) - int32(b[i])
		sum += diff * diff
	}

	return sum
}

func toUint8(v float32) uint8 {
	return uint8(clamp(v) * 255.0)
}

func Vectorize(
	req *TransactionRequest,
	norm Normalization,
	mccRisk map[string]float32,
) [14]uint8 {
	var vec [14]uint8
	vec[0] = toUint8(req.Transaction.Amount / norm.MaxAmount)
	vec[1] = toUint8(
		float32(req.Transaction.Installments) /
			norm.MaxInstallments,
	)

	if req.Customer.AvgAmount > 0 {
		ratio := (req.Transaction.Amount / req.Customer.AvgAmount) / norm.AmountVsAvgRatio
		vec[2] = toUint8(ratio)
	}

	t, _ := time.Parse(
		time.RFC3339,
		req.Transaction.RequestedAt,
	)

	vec[3] = toUint8(float32(t.UTC().Hour()) / 23.0)

	weekday := int(t.UTC().Weekday())
	weekday = (weekday + 6) % 7

	vec[4] = toUint8(float32(weekday) / 6.0)

	if req.LastTransaction == nil {
		// uint8 não permite -1, então mapeamos a ausência para 0
		vec[5] = 0
		vec[6] = 0
	} else {
		lastTime, _ := time.Parse(
			time.RFC3339,
			req.LastTransaction.RequestedAt,
		)

		minutes := float32(
			t.Sub(lastTime).Minutes(),
		)

		vec[5] = toUint8(minutes / norm.MaxMinutes)

		vec[6] = toUint8(
			req.LastTransaction.KmFromCurrent /
				norm.MaxKm,
		)
	}

	vec[7] = toUint8(
		req.Terminal.KmFromHome / norm.MaxKm,
	)

	vec[8] = toUint8(
		float32(req.Customer.TxCount24h) /
			norm.MaxTxCount24h,
	)

	if req.Terminal.IsOnline {
		vec[9] = 255 // Mapeado para o valor máximo em vez de 1
	}

	if req.Terminal.CardPresent {
		vec[10] = 255
	}

	known := false

	for _, merchant := range req.Customer.KnownMerchants {
		if merchant == req.Merchant.ID {
			known = true
			break
		}
	}

	if !known {
		vec[11] = 255
	}

	risk, ok := mccRisk[req.Merchant.MCC]

	if !ok {
		risk = 0.5
	}

	vec[12] = toUint8(risk)

	vec[13] = toUint8(
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
