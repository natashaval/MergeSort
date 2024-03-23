#include <mpi.h>
#include <random>
#include <algorithm>
#include <math.h>

using namespace std;

// Source: https://github-pages.ucl.ac.uk/research-computing-with-cpp/09distributed_computing/sec02ProgrammingWithMPI.html
// Task is to merge the process,
// proc 0 <- proc 1; proc 2 <- proc 3; proc 4 <- proc 5; proc 6 <- proc 7
// proc 0 <- proc 2; proc 4 <- proc 6
// proc 0 <- proc 4;
// merge two lists which are stored next to one another in a buffer
// https://selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/mergeSort/mergeSort.html
void merge(double *buffer, int start1, int size1, int size2)
{
    double *working_buffer = new double[size1 + size2];
    int count1 = 0;
    int count2 = 0;
    int start2 = start1 + size1;
    while ((count1 < size1) & (count2 < size2))
    {
        double x1 = buffer[start1 + count1];
        double x2 = buffer[start2 + count2];
        if (x1 < x2)
        {
            working_buffer[count1 + count2] = x1;
            count1++;
        }
        else
        {
            working_buffer[count1 + count2] = x2;
            count2++;
        }
    }

    // Fill buffer with whichever values remain
    for (int i = count1; i < size1; i++)
    {
        working_buffer[i + count2] = buffer[start1 + i];
    }
    for (int i = count2; i < size2; i++)
    {
        working_buffer[count1 + i] = buffer[start2 + i];
    }

    for (int i = 0; i < (size1 + size2); i++)
    {
        buffer[i] = working_buffer[i];
    }

    delete[] working_buffer;
}

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);

    int process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    int num_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    if (process_id == 0)
    {
        const int N = 256;
        std::mt19937_64 rng;
        std::uniform_real_distribution<double> dist(0, 1);
        double master_list[N];
        for (int i = 0; i < N; i++)
        {
            master_list[i] = dist(rng);
        }

        int listSize = N / num_proc; // We are using a multiple of four to avoid dealing with remainders!

        // Send the list data in messages
        for (int i = 1; i < num_proc; i++)
        {
            double *buffer_start = master_list + listSize * i;
            MPI_Send(buffer_start,
                     listSize,
                     MPI_DOUBLE,
                     i,
                     0,
                     MPI_COMM_WORLD);
        }

        std::sort(master_list, master_list + listSize);

        for (int i = 1; i < num_proc; i++)
        {
            double *buffer_start = master_list + listSize * i; // copy received buffers back into master list
            // we just need the sorted sublists, but we don't care which process is sending them so we'll take
            // them in whichever order they come using MPI_ANY_SOURCE. The loop makes sure that receive enough
            // messages.

            MPI_Recv(buffer_start,
                     listSize,
                     MPI_DOUBLE,
                     MPI_ANY_SOURCE,
                     1,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }

        // Merge all the lists
        // Again we'll cheat this loop a bit by assuming that num_proc and N are both powers of two
        // In real code we would have to deal with things like remainders properly but this example is already quite big!
        // LEAVE THIS MERGE FUNCTION AS IT IS, this is to merge the array after receiving it
        for (int i = num_proc; i > 1; i /= 2)
        {
            int subListSize = N / i;
            for (int j = 0; j < i; j += 2)
            {
                merge(master_list, j * subListSize, subListSize, subListSize);
            }
        }

        printf("Sorted List: ");
        for (int i = 0; i < N; i++)
        {
            printf("%f ", master_list[i]);
        }
        printf("\n");
    }
    else
    {
        int listSize = 256 / num_proc; // I am cheating here because I don't want to send another message communicating the size in this simple example.
        double sub_list[listSize];     // Only needs to be big enough to hold our sub list
        MPI_Recv(sub_list,
                 listSize,
                 MPI_DOUBLE,
                 0,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        printf("Process %d received a list starting with %f\n", process_id, sub_list[0]);

        std::sort(sub_list, sub_list + listSize);
        // MOVE THE CLOSING } after MPI_Recv to receive the initial data from scattering above
    }

    // TODO: MPI_REDUCE (?) to combine process for each round; see L8 - L11
    // round 0: receive i % 2 = 0; send (i-1)%2 = 0
    // round 1: receive (i/2) % 2 = 0; send (i-1)/2 % 2 = 0;
    // round r: receive (i/2^r)%2 = 0; send (i-1)/2^r % 2 = 0;
    // DO FOR LOOP for each round log2 of proc_num -> and define who should send and receive based on i/2^r calculation

    // REFER illustration: https://selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/mergeSort/mergeSort.html#
    // but this one use recursive instead of iterative, what we want is the iteration
    int count = 0;
    double *buffer;
    for (int round = 0; round < log2(num_proc); round++)
    {
        int source_rank = (process_id - 1) / pow(2, round);
        int recv_rank = process_id / (pow(2, round));
        if (recv_rank % 2 == 0)
        {
            MPI_Status msg_status;  // variable for holding message status
            // count to determine the size of array to be receive from previous processor
            MPI_Probe(process_id, 1, MPI_COMM_WORLD, &msg_status);
            // set buffer size appropriately
            buffer = new double(count);
            // receive message
            MPI_Recv(buffer, count, MPI_DOUBLE, source_rank, 1, MPI_COMM_WORLD, &msg_status);
        }
        else if (source_rank % 2 == 0)
        {
            MPI_Send(buffer, count, MPI_DOUBLE, recv_rank, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
}
