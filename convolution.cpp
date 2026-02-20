#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <mpi.h>

//differnt sizes to test with
const int IMAGE_HEIGHT = 1024;
const int IMAGE_WIDTH = 1024;
//the total number of filters 
const int K_FILTERS = 128;
const int FILTER_HEIGHT = 5; 
const int FILTER_WIDTH = 5;

//this calculates the sizes output sizes along with the image and filter size
const int OUTPUT_HEIGHT = IMAGE_HEIGHT - FILTER_HEIGHT + 1;
const int OUTPUT_WIDTH = IMAGE_WIDTH - FILTER_WIDTH + 1;
const int IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
const int FILTER_SIZE = FILTER_HEIGHT * FILTER_WIDTH;
const int OUTPUT_MAP_SIZE = OUTPUT_HEIGHT * OUTPUT_WIDTH;


//this performs a 2D convolution on one image with one filter. All data is in flat 1D vectors for the performance

std::vector<double> serial_convolution_single(const std::vector<double>& image,
                                              const std::vector<double>& filter) 
{
    std::vector<double> output_map(OUTPUT_MAP_SIZE);

    //This loops over output map rows
    for (int y = 0; y < OUTPUT_HEIGHT; ++y) {
        //This loops over output map cols
        for (int x = 0; x < OUTPUT_WIDTH; ++x) { 
            
            double sum = 0.0;
            //This loops over filter rows
            for (int j = 0; j < FILTER_HEIGHT; ++j) {
                //This loops over filter cols
                for (int i = 0; i < FILTER_WIDTH; ++i) { 
                    
                    //this finds the pixels in the image
                    int img_y = y + j;
                    int img_x = x + i;
                    int img_idx = img_y * IMAGE_WIDTH + img_x;
                    int filter_idx = j * FILTER_WIDTH + i;
                    sum += image[img_idx] * filter[filter_idx];
                }
            }
            output_map[y * OUTPUT_WIDTH + x] = sum;
        }
    }
    return output_map;
}

// this verifies that the serial and parallel outputs are similar and since the floating point isn't perfect there is small error of margin that I allow
bool verify(const std::vector<double>& serial, const std::vector<double>& parallel, double epsilon = 1e-6) {
    if (serial.size() != parallel.size()) {
        std::cout << "Verification FAILED: Sizes do not match!" << std::endl;
        return false;
    }
    for (size_t i = 0; i < serial.size(); ++i) {
        if (std::fabs(serial[i] - parallel[i]) > epsilon) {
            std::cout << "Verification FAILED: Mismatch at index " << i
                      << " Serial: " << serial[i] << " Parallel: " << parallel[i]
                      << " Diff: " << std::fabs(serial[i] - parallel[i]) << std::endl;
            return false;
        }
    }
    return true;
}


int main(int argc, char** argv) {
    
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //This is for data allocation so just like before all the data is held in flat 1D vectors so rank 0 holds all data
    std::vector<double> image;
    std::vector<double> all_filters;
    std::vector<double> serial_output;
    std::vector<double> parallel_output;

    //the root process initializes all data
    if (world_rank == 0) {
        std::cout << "--- Parallel 2D Convolution ---" << std::endl;
        std::cout << "Running with " << world_size << " processes." << std::endl;
        std::cout << "Image: " << IMAGE_HEIGHT << "x" << IMAGE_WIDTH << std::endl;
        std::cout << "Filters: " << K_FILTERS << " (" << FILTER_HEIGHT << "x" << FILTER_WIDTH << ")" << std::endl;
        
        image.resize(IMAGE_SIZE);
        all_filters.resize(K_FILTERS * FILTER_SIZE);
        serial_output.resize(K_FILTERS * OUTPUT_MAP_SIZE);
        parallel_output.resize(K_FILTERS * OUTPUT_MAP_SIZE);

        //this initializes with random data
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (double& val : image) val = dis(gen);
        for (double& val : all_filters) val = dis(gen);
    }

    //this is serial baseline so it runs on the root 
    if (world_rank == 0) {
        std::cout << "\nRunning serial baseline on root..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<double> current_filter(FILTER_SIZE);
        
        for (int k = 0; k < K_FILTERS; ++k) {
            //this copies the k th filter from the flat array
            std::copy(all_filters.begin() + k * FILTER_SIZE,
                      all_filters.begin() + (k + 1) * FILTER_SIZE,
                      current_filter.begin());
            
            //this runs convolution for this single filter
            std::vector<double> one_map = serial_convolution_single(image, current_filter);

            //this cpies the result into the full serial output tensor
            std::copy(one_map.begin(), one_map.end(),
                      serial_output.begin() + k * OUTPUT_MAP_SIZE);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Serial Baseline Time: " << diff.count() << " seconds." << std::endl;
    }


    //this is parallel execution 
    
    //this starts timer for parallel run
    MPI_Barrier(MPI_COMM_WORLD); 
    double p_start = MPI_Wtime();

    //this is the broadcast image and the non root process has to allocate the space before the broadcast
    if (world_rank != 0) {
        image.resize(IMAGE_SIZE);
    }
    MPI_Bcast(image.data(), IMAGE_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- B. Calculate Scatterv Parameters ---
    // We need to divide K_FILTERS as evenly as possible among world_size processes.
    // `counts` = how many filters each process gets
    // `displs` = the *starting index* (in filters) for each process
    
    //this is how many filters each rank gets
    std::vector<int> counts(world_size); 
    std::vector<int> displs(world_size);
    
    int base_chunk = K_FILTERS / world_size;
    int remainder = K_FILTERS % world_size;
    int current_displ = 0;

    for (int i = 0; i < world_size; ++i) {
        counts[i] = base_chunk + (i < remainder ? 1 : 0);
        displs[i] = current_displ;
        current_displ += counts[i];
    }
    
    //each of the ranks needs to know their local number of filters
    int local_k = counts[world_rank];

    std::vector<int> counts_bytes(world_size);
    std::vector<int> displs_bytes(world_size);

    //for scattering the filters
    for (int i = 0; i < world_size; ++i) {
        counts_bytes[i] = counts[i] * FILTER_SIZE;
        displs_bytes[i] = displs[i] * FILTER_SIZE;
    }

    //now we scatter the filters and allocate the space for local filters
    std::vector<double> local_filters(local_k * FILTER_SIZE);
    
    //send and recive the requried data
    MPI_Scatterv(all_filters.data(),
                 counts_bytes.data(),
                 displs_bytes.data(),
                 MPI_DOUBLE,
                 local_filters.data(),
                 local_k * FILTER_SIZE,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    //this is for the local computation for the parrallel part each process works only on its local k filters
    std::vector<double> local_output(local_k * OUTPUT_MAP_SIZE);
    std::vector<double> current_filter(FILTER_SIZE);

    for (int k = 0; k < local_k; ++k) {
        //this gets the k th filter from our local buffer
        std::copy(local_filters.begin() + k * FILTER_SIZE,
                  local_filters.begin() + (k + 1) * FILTER_SIZE,
                  current_filter.begin());

        //this runs the convolution
        std::vector<double> one_map = serial_convolution_single(image, current_filter);

        //this copies the result to the local output buffer
        std::copy(one_map.begin(), one_map.end(),
                  local_output.begin() + k * OUTPUT_MAP_SIZE);
    }

    //this gather results from the above tests we need to recalculate counts displays for the output data size
    for (int i = 0; i < world_size; ++i) {
        counts_bytes[i] = counts[i] * OUTPUT_MAP_SIZE;
        displs_bytes[i] = displs[i] * OUTPUT_MAP_SIZE;
    }

    //this recives the buffers and counts 
    MPI_Gatherv(local_output.data(),
                local_k * OUTPUT_MAP_SIZE, 
                MPI_DOUBLE,
                parallel_output.data(),
                counts_bytes.data(),
                displs_bytes.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    //this stops the timer and verify 
    MPI_Barrier(MPI_COMM_WORLD); 
    double p_end = MPI_Wtime();

    if (world_rank == 0) {
        double parallel_time = p_end - p_start;
        double serial_time = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - 
            std::chrono::high_resolution_clock::now()
        ).count(); 
        
        //this gets serial time 
        auto start = std::chrono::high_resolution_clock::now();
    }
    
    //printing out for the terminal for verfication 
    if (world_rank == 0) {
        double parallel_time = p_end - p_start;
        std::cout << "\nParallel Run Time: " << parallel_time << " seconds." << std::endl;
        
        std::cout << "\nVerifying results..." << std::endl;
        if (verify(serial_output, parallel_output)) {
            std::cout << "Verification: SUCCESS! Serial and Parallel outputs match." << std::endl;
        } else {
            std::cout << "Verification: FAILED! Outputs do not match." << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}