/*
* Adaptado de: http://w...content-available-to-author-only...s.org/sieve-of-eratosthenes
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>

int sieveOfEratosthenes(int n)
{
   // Cria um array booleano "prime[0..n]" e inicializa
   // todos os elementos como "true". Um valor em prime[i]
   // será "false" se i não for primo; caso contrário, permanece "true".
   int primes = 0; 
   bool *prime = (bool*) malloc((n+1) * sizeof(bool));
   int sqrt_n = sqrt(n);
     
   memset(prime, true, (n+1) * sizeof(bool));
   
   int i, p;
   
   #pragma omp parallel for
   for (p = 2; p <= sqrt_n; p++)
   {
       // Se prime[p] ainda for "true", então p é primo
       if (prime[p] == true)
       {
           // Atualiza todos os múltiplos de p
           #pragma omp parallel for
           for (i = p*2; i <= n; i += p)
               prime[i] = false;
       }
   }
   
   // Conta a quantidade de números primos
   #pragma omp parallel for reduction(+:primes)
   for (int p = 2; p <= n; p++)
       if (prime[p])
           primes++;
 
   free(prime);  // Libera a memória alocada
   return primes;
}
     
int main()
{
   int n = 100000000;
   printf("%d\n", sieveOfEratosthenes(n));
   return 0;
}
