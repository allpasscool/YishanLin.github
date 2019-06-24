package mapreduce

import (
	"fmt"
	"sync"
)

//
// schedule() starts and waits for all tasks in the given phase (mapPhase
// or reducePhase). the mapFiles argument holds the names of the files that
// are the inputs to the map phase, one per map task. nReduce is the
// number of reduce tasks. the registerChan argument yields a stream
// of registered workers; each item is the worker's RPC address,
// suitable for passing to call(). registerChan will yield all
// existing registered workers (if any) and new ones as they register.
//
func schedule(jobName string, mapFiles []string, nReduce int, phase jobPhase, registerChan chan string) {
	var ntasks int
	var n_other int // number of inputs (for reduce) or outputs (for map)
	switch phase {
	case mapPhase:
		ntasks = len(mapFiles)
		n_other = nReduce
	case reducePhase:
		ntasks = nReduce
		n_other = len(mapFiles)
	}

	fmt.Printf("Schedule: %v %v tasks (%d I/Os)\n", ntasks, phase, n_other)

	var wg sync.WaitGroup
	if phase == mapPhase {
		for i, j := range mapFiles {

			arg := DoTaskArgs{jobName, j, phase, i, n_other}
			var reply ShutdownReply
			var ok bool

			wg.Add(1)
			go func() {
				go func() {
					v, _ := <-registerChan
					ok = call(v, "Worker.DoTask", arg, &reply)

					if ok == false {
						var wg1 sync.WaitGroup
						ok1 := false

						for ok1 == false {
							v, _ = <-registerChan

							wg1.Add(1)
							go func() {
								ok1 = call(v, "Worker.DoTask", arg, &reply)
								wg1.Done()
								wg1.Wait()
							}()
							wg1.Wait()
							if ok1 == true {
								go func() { registerChan <- v }()
								wg1.Wait()
								break
							}
						}
						wg.Done()
						wg1.Wait()
						wg.Wait()
					} else {
						go func() { registerChan <- v }()
						wg.Done()
						wg.Wait()
					}
				}()
				wg.Wait()
			}()
		}
	} else if phase == reducePhase {

		for i := 0; i < nReduce; i++ {
			arg := DoTaskArgs{jobName, "", phase, i, n_other}
			var reply ShutdownReply
			wg.Add(1)
			var ok bool
			go func() {
				go func() {
					v, _ := <-registerChan
					ok = call(v, "Worker.DoTask", arg, &reply)
					if ok == false {
						var wg1 sync.WaitGroup
						ok1 := false
						for ok1 == false {
							v, _ = <-registerChan
							wg1.Add(1)
							go func() {
								ok1 = call(v, "Worker.DoTask", arg, &reply)
								wg1.Done()
								wg1.Wait()
							}()
							wg1.Wait()
							if ok1 == true {
								go func() { registerChan <- v }()
								wg1.Wait()
								break
							}
						}
						wg.Done()
						wg1.Wait()
						wg.Wait()
					} else {
						var wg2 sync.WaitGroup
						wg2.Add(1)
						go func() {
							registerChan <- v
							wg2.Done()
						}()
						wg.Done()
						wg.Wait()
					}
				}()
				wg.Wait()
			}()
		}
		wg.Wait()
	}
	wg.Wait()
	wg.Wait()
	wg.Wait()
	wg.Wait()

	// All ntasks tasks have to be scheduled on workers. Once all tasks
	// have completed successfully, schedule() should return.
	//
	// Your code here (Part 2, 2B).
	//
	fmt.Printf("Schedule: %v done\n", phase)
}
