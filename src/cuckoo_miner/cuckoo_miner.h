// Copyright 2017 The Grin Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Functions specific to cuckoo-miner's modifications, localising as many
// changes as possible here to avoid diverging from the original source

#ifndef CUCKOO_MINER_H
#define CUCKOO_MINER_H

#include <stdio.h>
#include "hash_impl.h"

#define SQUASH_OUTPUT 0

#if SQUASH_OUTPUT
#define printf(fmt, ...) (0)
#endif

/**
 * Replace the SHA256 function with our own, pulled from secp256k
 */

void SHA256(unsigned char * in, u32 len, unsigned char* out){
    secp256k1_sha256_t sha;
    secp256k1_sha256_initialize(&sha);
    secp256k1_sha256_write(&sha, in, len);
    secp256k1_sha256_finalize(&sha, out);
}

//Handy function to keep around for debugging
static void print_buf(const char *title, const unsigned char *buf, size_t buf_len)
{
    size_t i = 0;
    printf("%s\n", title);
    for(i = 0; i < buf_len; ++i)
    printf("%02X%s", buf[i],
             ( i + 1 ) % 16 == 0 ? "\r\n" : " " );

}

#endif //CUCKOO_MINER_H

