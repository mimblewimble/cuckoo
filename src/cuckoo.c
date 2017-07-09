// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2016 John Tromp

#include "cuckoo.h"
#include <inttypes.h> // for SCNx64 macro
#include <stdio.h>    // printf/scanf
#include <stdlib.h>   // exit
#include <unistd.h>   // getopt
#include <assert.h>   // d'uh

int main(int argc, char **argv) {
  unsigned char header[32];
  int nonce = 0;
  int c;

  char *hexstring = "A6C16443FC82250B49C7FAA3876E7AB89BA687918CB00C4C10D6625E3A2E7BCC";
  int i;
  uint8_t str_len = strlen(hexstring);

  for (i = 0; i < (str_len / 2); i++) {
      sscanf(hexstring + 2*i, "%02x", &header[i]);
  }

  /*while ((c = getopt (argc, argv, "h:n:")) != -1) {
    switch (c) {
      case 'h':
        header = optarg;
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
    }
  }*/
  printf("Verifying size %d proof for cuckoo%d(\"%s\",%d)\n",
               PROOFSIZE, EDGEBITS+1, header, nonce);
  /*char headernonce[32];
  u32 hdrlen = 32;
  memcpy(headernonce, header, hdrlen);
  memset(headernonce+hdrlen, 0, sizeof(headernonce)-hdrlen);
  ((u32 *)headernonce)[HEADERLEN/sizeof(u32)-1] = htole32(nonce);*/
  int sol_nonces[42]= {0x1, 0x11, 0x7a, 0xab, 0xee, 0x121, 0x154, 0x155,
                        0x1de, 0x1ec, 0x201, 0x226, 0x22b, 0x271, 0x2fa, 0x318,0x323,
                        0x32b, 0x33d, 0x33e, 0x3ab, 0x3d2, 0x43a, 0x43e, 0x46e, 0x4b4,
                        0x4ca, 0x4ce, 0x524, 0x549, 0x554, 0x587, 0x5a2, 0x612, 0x68a,
                        0x699, 0x6d8, 0x71b, 0x74e, 0x755, 0x760, 0x79c};
  
  int sol_index=0;
  for (int nsols=0; ;nsols++) {
    edge_t nonces[PROOFSIZE];
    for (int n = 0; n < PROOFSIZE; n++) {
      u64 nonce;
      nonces[n] = sol_nonces[sol_index++];
    }
    printf("header nonce length: %d\n", sizeof(header));
    int pow_rc = verify(nonces, header, sizeof(header));
    if (pow_rc == POW_OK) {
      printf("Verified with cyclehash ");
      unsigned char cyclehash[32];
      SHA256((unsigned char *)nonces, sizeof(nonces), cyclehash);
      for (int i=0; i<32; i++)
        printf("%02x", cyclehash[i]);
      printf("\n");
    } else {
      printf("FAILED due to %s\n", errstr[pow_rc]);
    }
    break;
  }
  return 0;
}
