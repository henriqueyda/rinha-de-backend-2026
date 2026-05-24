package main

import (
	"math/rand"
	"testing"
)

func mockIndexForBenchmark() *IVFIndex {
	clusters := make([]Cluster, KClusters)
	for c := 0; c < KClusters; c++ {
		var centroid [Dimensions]float32
		for d := 0; d < Dimensions; d++ {
			centroid[d] = rand.Float32()
		}
		clusters[c].Centroid = centroid

		// Simula 1000 vetores dentro de cada cluster
		clusters[c].Vectors = make([]Vector, 1000)
		for v := 0; v < 1000; v++ {
			var values [Dimensions]float32
			for d := 0; d < Dimensions; d++ {
				values[d] = rand.Float32()
			}
			clusters[c].Vectors[v] = Vector{
				Values: values,
				Fraud:  rand.Float32() > 0.8,
			}
		}
	}
	return &IVFIndex{Clusters: clusters}
}

func BenchmarkSearch(b *testing.B) {
	idx := mockIndexForBenchmark()
	var query [Dimensions]float32
	for d := 0; d < Dimensions; d++ {
		query[d] = rand.Float32()
	}

	b.ResetTimer() // Ignora o tempo gasto criando o mock acima

	for i := 0; i < b.N; i++ {
		_, _ = idx.Search(query)
	}
}
