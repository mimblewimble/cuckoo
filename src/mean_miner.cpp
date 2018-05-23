// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2018 John Tromp

#include "mean_miner.hpp"
#include <unistd.h>
#include <sys/time.h>

#include "cuckoo_miner/mean_miner_adds.h"

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

extern "C" int cuckoo_call(char* header_data, 
                           int header_length,
                           u32* sol_nonces){

	u64 start_time=timestamp();
  assert(NUM_THREADS_PARAM>0);
	NUM_TRIMS_PARAM = NUM_TRIMS_PARAM & -2;//Make even

  print_buf("(Mean Miner) Coming in is: ", (const unsigned char*) header_data, header_length);
  printf("Num Trims %d\n", NUM_TRIMS_PARAM);
  u32 range = 1;
#ifdef SAVEEDGES
  bool showcycle = 1;
#else
  bool showcycle = 0;
#endif
  struct timeval time0, time1;
  u32 timems;
  //char header[HEADERLEN];
  //u32 len;
  bool allrounds = false;
  //int c;
/*
  memset(header, 0, sizeof(header));
  while ((c = getopt (argc, argv, "ah:m:n:r:st:x:")) != -1) {
    switch (c) {
      case 'a':
        allrounds = true;
        break;
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
        ntrims = atoi(optarg) & -2; // make even as required by solve()
        break;
      case 's':
        showcycle = true;
        break;
      case 't':
        nthreads = atoi(optarg);
        break;
    }
  }
  printf("Looking for %d-cycle on cuckoo%d(\"%s\",%d", PROOFSIZE, NODEBITS, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with 50%% edges\n");
  */
  solver_ctx ctx(NUM_THREADS_PARAM, NUM_TRIMS_PARAM, allrounds, showcycle);
  ctx.sols.reserve(10*PROOFSIZE);
  u64 sbytes = ctx.sharedbytes();
  u32 tbytes = ctx.threadbytes();
  int sunit,tunit;
  for (sunit=0; sbytes >= 10240; sbytes>>=10,sunit++) ;
  for (tunit=0; tbytes >= 10240; tbytes>>=10,tunit++) ;
  printf("Using %d%cB bucket memory at %lx,\n", sbytes, " KMGT"[sunit], (u64)ctx.trimmer->buckets);
  printf("%dx%d%cB thread memory at %lx,\n", NUM_THREADS_PARAM, tbytes, " KMGT"[tunit], (u64)ctx.trimmer->tbuckets);
  printf("%d-way siphash, and %d buckets.\n", NSIPHASH, NX);

  u32 sumnsols = 0;
  for (u32 r = 0; r < range; r++) {
    gettimeofday(&time0, 0);
    //ctx.setheadernonce(header, sizeof(header), nonce + r);
    ctx.setheadergrin(header_data, header_length);
    printf("k0 k1 %lx %lx\n", ctx.trimmer->sip_keys.k0, ctx.trimmer->sip_keys.k1);
    u32 nsols = ctx.solve();
    gettimeofday(&time1, 0);
    timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    printf("Time: %d ms\n", timems);

    for (unsigned s = 0; s < nsols; s++) {
      printf("Solution");
      u32* prf = &ctx.sols[s * PROOFSIZE];
      for (u32 i = 0; i < PROOFSIZE; i++){
        printf(" %jx", (uintmax_t)prf[i]);
        sol_nonces[i] = prf[i];
      }
      printf("\n");
      int pow_rc = verify(prf, &ctx.trimmer->sip_keys);
      if (pow_rc == POW_OK) {
        printf("Verified with cyclehash ");
        unsigned char cyclehash[32];
        blake2b((void *)cyclehash, sizeof(cyclehash), (const void *)prf, sizeof(proof), 0, 0);
        for (int i=0; i<32; i++)
          printf("%02x", cyclehash[i]);
        printf("\n");
        if(SINGLE_MODE){
         update_stats(start_time);
        }
        return 1;
      } else {
        printf("FAILED due to %s\n", errstr[pow_rc]);
      }
    }
    sumnsols += nsols;
  }
  printf("%d total solutions\n", sumnsols);
  if (SINGLE_MODE) {
    update_stats(start_time);
  }
  return 0;
}
