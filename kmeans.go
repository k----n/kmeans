// Package kmeans implements the k-means clustering algorithm
// See: https://en.wikipedia.org/wiki/K-means_clustering
package kmeans

import (
	"fmt"
	"math/rand"
	"sync/atomic"
	"sync"

	"github.com/k----n/clusters"
	"github.com/k----n/classifier/parallel"
)

// Kmeans configuration/option struct
type Kmeans struct {
	// number of threads
	Threads int
	// when a plotter is set, Plot gets called after each iteration
	plotter Plotter
	// deltaThreshold (in percent between 0.0 and 0.1) aborts processing if
	// less than n% of data points shifted clusters in the last iteration
	deltaThreshold float64
	// iterationThreshold aborts processing when the specified amount of
	// algorithm iterations was reached
	iterationThreshold int
}

// The Plotter interface lets you implement your own plotters
type Plotter interface {
	Plot(cc clusters.Clusters, iteration int) error
}

// NewWithOptions returns a Kmeans configuration struct with custom settings
func NewWithOptions(deltaThreshold float64, plotter Plotter) (Kmeans, error) {
	if deltaThreshold <= 0.0 || deltaThreshold >= 1.0 {
		return Kmeans{}, fmt.Errorf("threshold is out of bounds (must be >0.0 and <1.0, in percent)")
	}

	return Kmeans{
		plotter:            plotter,
		deltaThreshold:     deltaThreshold,
		iterationThreshold: 96,
	}, nil
}

// New returns a Kmeans configuration struct with default settings
func New() Kmeans {
	m, _ := NewWithOptions(0.01, nil)
	return m
}

// Partition executes the k-means algorithm on the given dataset and
// partitions it into k clusters
func (m Kmeans) Partition(dataset clusters.Observations, k int) (clusters.Clusters, error) {
	if k > len(dataset) {
		return clusters.Clusters{}, fmt.Errorf("the size of the data set must at least equal k")
	}

	cc, err := clusters.New(k, dataset)
	if err != nil {
		return cc, err
	}

	points := make([]int, len(dataset))
	var changes atomic.Uint64
	changes.Add(1)

	for i := 0; changes.Load() > 0; i++ {
		changes.Store(0)
		cc.ResetThreads(m.Threads)
		var mut [256]sync.RWMutex

		parallel.ForEach(len(dataset), m.Threads, func (p int) {
			point := dataset[p]
			for i := range mut {
				mut[i].RLock()
			}
			ci := cc.Nearest(point)
			for i := range mut {
				mut[i].RUnlock()
			}
			mut[ci & 255].Lock()
			cc[ci].Append(point)
			if points[p] != ci {
				points[p] = ci
				changes.Add(1)
			}
			mut[ci & 255].Unlock()
		})

		parallel.ForEach(len(cc), m.Threads, func (ci int) {
			if len(cc[ci].Observations) == 0 {
				// During the iterations, if any of the cluster centers has no
				// data points associated with it, assign a random data point
				// to it.
				// Also see: http://user.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html
				var ri int
				for {
					// find a cluster with at least two data points, otherwise
					// we're just emptying one cluster to fill another
					ri = rand.Intn(len(dataset)) //nolint:gosec // rand.Intn is good enough for this
					mut[ri & 255].RLock()
					if len(cc[points[ri]].Observations) > 1 {
						mut[ri & 255].RUnlock()
						break
					} else {
						mut[ri & 255].RUnlock()
					}
				}
				mut[ci & 255].Lock()
				cc[ci].Append(dataset[ri])
				points[ri] = ci
				mut[ci & 255].Unlock()

				// Ensure that we always see at least one more iteration after
				// randomly assigning a data point to a cluster
				changes.Add(uint64(len(dataset)))
			}
		})

		if changes.Load() > 0 {
			cc.RecenterThreads(m.Threads)
		}
		if m.plotter != nil {
			err := m.plotter.Plot(cc, -int(changes.Load()))
			if err != nil {
				return nil, fmt.Errorf("failed to plot chart: %s", err)
			}
		}
		if i == m.iterationThreshold ||
			int(changes.Load()) < int(float64(len(dataset))*m.deltaThreshold) {
			// fmt.Println("Aborting:", changes, int(float64(len(dataset))*m.TerminationThreshold))
			break
		}
	}

	return cc, nil
}
