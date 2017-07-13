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
#include <string.h>
#include "hash_impl.h"

#define SQUASH_OUTPUT 1

#if SQUASH_OUTPUT
#define printf(fmt, ...) (0)
#endif

/** 
 * Some hardwired stuff to hold properties
 * without dynamically allocating memory
 * kept very simple for now
 */

#define MAX_NUM_PROPERTIES 16
#define MAX_PROPERTY_NAME_LENGTH 64
#define MAX_PROPERTY_DESC_LENGTH 256

int allocated_properties=0;

struct PLUGIN_PROPERTY {
    char name[MAX_PROPERTY_NAME_LENGTH];
    char description[MAX_PROPERTY_DESC_LENGTH];
    u32 default_value;
    u32 min_value;
    u32 max_value;
};

PLUGIN_PROPERTY PROPS[MAX_NUM_PROPERTIES];

void add_plugin_property(PLUGIN_PROPERTY new_property){
    if (allocated_properties>MAX_NUM_PROPERTIES-1){
        return;
    }
    PROPS[allocated_properties++]=new_property;
}

/*
 * Either fills given string with properties, or returns error code
 * if there isn't enough buffer
 */

int get_properties_as_json(char* prop_string, int* length){
    int remaining=*length;
    const char* property_json = "{\"name\":\"%s\",\"description\":\"%s\",\"default_value\":%d,\"min_value\":%d,\"max_value\":%d}";
    //minimum return is "[]\0"
    if (remaining<=3){
        //TODO: Meaningful return code
        return -1;
    }
    prop_string[0]='[';
    int last_write_pos=1;
    for (int i=0;i<allocated_properties;i++){
        int last_written=snprintf(prop_string+last_write_pos, 
                              remaining, 
                              property_json, PROPS[i].name, 
                              PROPS[i].description, PROPS[i].default_value,
                              PROPS[i].min_value, PROPS[i].max_value);
        remaining-=last_written;
        last_write_pos+=last_written;
        //no room for anything else, comma or trailing ']'
        if (remaining<2){
            //TODO: meaningful error code
            return -1;
        }
        //write comma
        if (i<allocated_properties-1){
            //overwrite trailing \0 in this case
            prop_string[last_write_pos++]=',';
        } 
    }
    //write final characters
    if (remaining<2){
        return -1;
    }
    //overwrite trailing \0
    prop_string[last_write_pos]=']';
    prop_string[last_write_pos+1]='\0';
    remaining -=2;
    *length=(*length)-remaining+1;
    
    //empty set
    if (*length==3){
        *length=2;
    }
}

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

