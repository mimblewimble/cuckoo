// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2017 John Tromp

#include "mean_miner.hpp"
#include <unistd.h>
#include <sys/time.h>

extern "C" int cuckoo_call(char* header_data, 
                           int header_length, 
                           int ntrims,
                           int nthreads,
                           u32* sol_nonces){
  
  char header[HEADERLEN];
  int c;
  int nonce = 0;
  int range = 1;
  struct timeval time0, time1;
  u32 timems;

  if (ntrims<=0) ntrims==1;

  assert(nthreads > 1);

  assert(header_length <= sizeof(header));

  memset(header, 0, sizeof(header));

  memcpy(header, header_data, header_length);

  
  /*while ((c = getopt (argc, argv, "h:m:n:r:t:x:")) != -1) {
    switch (c) {
      case 'h':
        len = strlen(optarg);
        assert(len <= sizeof(header));
        memcpy(header, optarg, len);
        break;
      case 'x':
        len = strlen(optarg)/2;
        assert(len == sizeof(header));
        for (u32 i=0; i<len; i++)
          sscanf(optarg+2*i, "%2hhx", header+i);
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'r':
        range = atoi(optarg);
        break;
      case 'm':
        ntrims = atoi(optarg);
        break;
      case 't':
        nthreads = atoi(optarg);
        break;
    }
  }*/
  printf("Looking for %d-cycle on cuckoo%d(\"%s\",%d", PROOFSIZE, EDGEBITS+1, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with 50%% edges\n");

  solver_ctx ctx(nthreads, ntrims);

  u32 sbytes = ctx.sharedbytes();
  u32 tbytes = ctx.threadbytes();
  int sunit,tunit;
  for (sunit=0; sbytes >= 10240; sbytes>>=10,sunit++) ;
  for (tunit=0; tbytes >= 10240; tbytes>>=10,tunit++) ;
  printf("Using %d%cB bucket memory at %lx,\n", sbytes, " KMGT"[sunit], (u64)ctx.trimmer->buckets);
  printf("%dx%d%cB thread memory at %lx,\n", nthreads, tbytes, " KMGT"[tunit], (u64)ctx.trimmer->tbuckets);
  printf("%d-way siphash, and %d buckets.\n", NSIPHASH, NBUCKETS);

  thread_ctx *threads = (thread_ctx *)calloc(nthreads, sizeof(thread_ctx));
  assert(threads);

  u32 sumnsols = 0;
  for (u32 r = 0; r < range; r++) {
    gettimeofday(&time0, 0);
    //ctx.setheadernonce(header, sizeof(header), nonce + r);
    ctx.setheadergrin(header, sizeof(header), nonce + r);
    printf("k0 k1 %lx %lx\n", ctx.trimmer->sip_keys.k0, ctx.trimmer->sip_keys.k1);
    u32 nsols = ctx.solve();
    gettimeofday(&time1, 0);
    timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    printf("Time: %d ms\n", timems);

    for (unsigned s = 0; s < ctx.nsols; s++) {
      printf("Solution");
      for (u32 i = 0; i < PROOFSIZE; i++) {
        printf(" %jx", (uintmax_t)ctx.sols[s][i]);
        sol_nonces[i] = ctx.sols[s][i];
      }
      return 1;
      printf("\n");
    }
    sumnsols += nsols;
  }
  printf("%d total solutions\n", sumnsols);
  return 0;
}

extern "C" void cuckoo_description(char * name_buf,
                              int* name_buf_len,
                              char *description_buf,
                              int* description_buf_len){

  int ntrims   = 0;
  
  //TODO: check we don't exceed lengths.. just keep it under 256 for now
  int name_buf_len_in = *name_buf_len;
  const char* name = "cuckoo_mean_%d\0";
  sprintf(name_buf, name, EDGEBITS+1);
  *name_buf_len = strlen(name);
  
  const char* desc1 = "Looks for a %d-cycle on cuckoo%d with 50%% edges, using mean algorithm.\n\
Uses %d-way siphash and %d buckets.\0";

  sprintf(description_buf, desc1,     
  PROOFSIZE, EDGEBITS+1, NSIPHASH, NBUCKETS);
  *description_buf_len = strlen(description_buf);
 
}
