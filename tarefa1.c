#include <stdio.h>
#include <omp.h>

int main()
{
		int i;
        // set num of threads 
		#pragma omp parallel num_threads(2) 
		{	
            
        // read id of thread       
		int tid = omp_get_thread_num();  
		
		#pragma omp for ordered schedule(dynamic)
		for(i = 1; i <= 3; i++) 
		{
			#pragma omp ordered      
			printf("[PRINT1] T%d = %d \n",tid,i);
			printf("[PRINT2] T%d = %d \n",tid,i);
		}
	}
}

