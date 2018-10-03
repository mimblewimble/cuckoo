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

// Functions specific to cuda implementation
#ifndef CUDA_MEAN_ADDS_H
#define CUDA_MEAN_ADDS_H

#include "cuckoo_miner.h"

//forward dec
extern "C" int cuckoo_call(char* header_data, 
                           int header_length,
                           u32* sol_nonces);

pthread_mutex_t device_info_mutex = PTHREAD_MUTEX_INITIALIZER;

#define MAX_DEVICES 32
#define NUM_TUNE_PARAMS 9
u32 NUM_DEVICES=0;

typedef class cudaDeviceInfo {
  public:
    int device_id;
    int cuckoo_size;
    bool is_busy;
    cudaDeviceProp properties;
    //store the current hash rate
    u64 last_start_time;
    u64 last_end_time; 
    u64 last_solution_time;
    u32 iterations_completed;

    bool threw_error;
    int use_device_param;

    //Store parameters per device
    int tune_params[NUM_TUNE_PARAMS];

    cudaDeviceInfo();
    //Fill with default tuning parameters, ideally per device type
    void fill_tuning_params();
} CudaDeviceInfo;

cudaDeviceInfo::cudaDeviceInfo(){
    device_id=-1;
    cuckoo_size=EDGEBITS + 1;
    is_busy=false;
    last_start_time=0;
    last_end_time=0;
    last_solution_time=0;
    iterations_completed=0;
    threw_error=false;
		use_device_param=0;

    // just zero out tuning parameters
    for (int i=0;i<NUM_TUNE_PARAMS;i++){
        tune_params[i]=0;
    }
}

void cudaDeviceInfo::fill_tuning_params(){
    //TODO: Check device type and adjust to sensible defaults
    tune_params[0]=0; //expand
    tune_params[1]=176; //N_TRIMS
    tune_params[2]=4096; //GEN_A_BLOCKS
    tune_params[3]=256; //GEN_A_TPB
    tune_params[4]=128; //GEN_B_TPB
    tune_params[5]=512; //TRIM_TPB
    tune_params[6]=1024; //TAIL_TPB
    tune_params[7]=1024; //RECOVER_BLOCKS
    tune_params[8]=1024; //RECOVER_TPB
}

CudaDeviceInfo DEVICE_INFO[MAX_DEVICES];

void populate_device_info(){
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    DEVICE_INFO[i].device_id=i;
    cudaGetDeviceProperties(&DEVICE_INFO[i].properties, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", DEVICE_INFO[i].properties.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           DEVICE_INFO[i].properties.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           DEVICE_INFO[i].properties.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*DEVICE_INFO[i].properties.memoryClockRate*(DEVICE_INFO[i].properties.memoryBusWidth/8)/1.0e6);
  }
  NUM_DEVICES=nDevices;
}

/*
 * returns current stats for all working devices
 */

extern "C" int cuckoo_get_stats(char* prop_string, int* length){
    int remaining=*length;
    const char* device_stat_json = "{\"device_id\":\"%d\",\"cuckoo_size\":\"%d\",\"device_name\":\"%s\",\"in_use\":%d,\"has_errored\":%d,\"last_start_time\":%lld,\"last_end_time\":%lld,\"last_solution_time\":%lld,\"iterations_completed\":%d}";
    //minimum return is "[]\0"
    if (remaining<=3){
        //TODO: Meaningful return code
        return PROPERTY_RETURN_BUFFER_TOO_SMALL;
    }
    prop_string[0]='[';
    int last_write_pos=1;
    int devices=NUM_DEVICES;
		if (SINGLE_MODE){
			devices=1;
		}

    for (int i=0;i<devices;i++){
        int last_written=snprintf(prop_string+last_write_pos,
                              remaining,
                              device_stat_json, DEVICE_INFO[i].device_id, DEVICE_INFO[i].cuckoo_size,
                              DEVICE_INFO[i].properties.name, DEVICE_INFO[i].use_device_param, 
                              DEVICE_INFO[i].threw_error, DEVICE_INFO[i].last_start_time,
                              DEVICE_INFO[i].last_end_time, DEVICE_INFO[i].last_solution_time,
			      DEVICE_INFO[i].iterations_completed);
        remaining-=last_written;
        last_write_pos+=last_written;
        //no room for anything else, comma or trailing ']'
        if (remaining<2){
            //TODO: meaningful error code
            return PROPERTY_RETURN_BUFFER_TOO_SMALL;
        }
        //write comma
        if (i<devices-1){
            //overwrite trailing \0 in this case
            prop_string[last_write_pos++]=',';
        } 
    }
    //write final characters
    if (remaining<2){
        return PROPERTY_RETURN_BUFFER_TOO_SMALL;
    }
    //overwrite trailing \0
    prop_string[last_write_pos]=']';
    prop_string[last_write_pos+1]='\0';
    remaining -=2;
    *length=last_write_pos+1;
    
    //empty set
    if (*length==3){
        *length=2;
    }
    return PROPERTY_RETURN_OK;
}

u32 next_free_device_id(){
    for (int i=0;i<NUM_DEVICES;i++){
    	if (!DEVICE_INFO[i].is_busy&&!DEVICE_INFO[i].threw_error&&DEVICE_INFO[i].use_device_param){
	      return i;
		  }
    }
    return -1;
}

/**
 * Initialises all parameters, defaults, and makes them available
 * to a caller
 */

extern "C" int cuckoo_init(){
  populate_device_info();
	allocated_properties=0;

  PLUGIN_PROPERTY device_list_prop;
  strcpy(device_list_prop.name,"USE_DEVICE\0");
  strcpy(device_list_prop.description,"If set, include this device while mining parallel GPUs\0");
  device_list_prop.default_value=0;
  device_list_prop.min_value=0;
  device_list_prop.max_value=1;
  device_list_prop.is_per_device=true;
  add_plugin_property(device_list_prop);

  for (int i=0;i<NUM_TUNE_PARAMS;i++){
    PLUGIN_PROPERTY prop;
    switch (i) {
			 case 0: {strcpy(prop.name,"EXPAND\0"); prop.default_value = 0; break;}
			 case 1: {strcpy(prop.name,"N_TRIMS\0"); prop.default_value = 176; break;}
			 case 2: {strcpy(prop.name,"GEN_A_BLOCKS\0"); prop.default_value = 4096; break;}
			 case 3: {strcpy(prop.name,"GEN_A_TPB\0"); prop.default_value = 256; break;}
			 case 4: {strcpy(prop.name,"GEN_B_TPB\0"); prop.default_value = 128; break;}
			 case 5: {strcpy(prop.name,"TRIM_TPB\0"); prop.default_value = 512; break;}
			 case 6: {strcpy(prop.name,"TAIL_TPB\0"); prop.default_value = 1024; break;}
			 case 7: {strcpy(prop.name,"RECOVER_BLOCKS\0"); prop.default_value = 1024; break;}
			 case 8: {strcpy(prop.name,"RECOVER_TPB\0"); prop.default_value = 1024; break;}
		}
    strcpy(prop.description,"Tuning Parameter\0");
    prop.min_value=1;
    prop.max_value=1024;
    prop.is_per_device=true;
    add_plugin_property(prop);
  }

  for (int i=0;i<NUM_DEVICES;i++){
     DEVICE_INFO[i].use_device_param = device_list_prop.default_value;
     //only use device 0 by default, will need to specify explicitly otherwise
     if (i==0) DEVICE_INFO[i].use_device_param=1;

     DEVICE_INFO[i].fill_tuning_params();
  }

  return PROPERTY_RETURN_OK;
}

/// Return a simple json list of parameters

extern "C" int cuckoo_parameter_list(char *params_out_buf,
                                     int* params_len){
  return get_properties_as_json(params_out_buf, params_len);
  
                                  
}

/// Set a parameter

extern "C" int cuckoo_set_parameter(char *param_name,
                                     int param_name_len,
                                     int device_id,
                                     int value){
  
  if (param_name_len > MAX_PROPERTY_NAME_LENGTH) return PROPERTY_RETURN_TOO_LONG;
  if (device_id > NUM_DEVICES-1) return PROPERTY_RETURN_INVALID_DEVICE;
  char compare_buf[MAX_PROPERTY_NAME_LENGTH];
  snprintf(compare_buf,param_name_len+1,"%s", param_name);
  if (strcmp(compare_buf,"USE_DEVICE")==0){
    if (value>=PROPS[0].min_value && value<=PROPS[0].max_value){
       DEVICE_INFO[device_id].use_device_param=value;
       return PROPERTY_RETURN_OK;
    } else {
      return PROPERTY_RETURN_OUTSIDE_RANGE;
    }
  }
	if (strcmp(compare_buf,"EXPAND")==0){
    if (value>=PROPS[1].min_value && value<=PROPS[1].max_value){
       DEVICE_INFO[device_id].tune_params[0]=value;
       return PROPERTY_RETURN_OK;
    } else {
      return PROPERTY_RETURN_OUTSIDE_RANGE;
    }
  }
  if (strcmp(compare_buf,"N_TRIMS")==0){
    if (value>=PROPS[2].min_value && value<=PROPS[2].max_value){
       DEVICE_INFO[device_id].tune_params[1]=value;
       return PROPERTY_RETURN_OK;
    } else {
      return PROPERTY_RETURN_OUTSIDE_RANGE;
    }
  }
  if (strcmp(compare_buf,"GEN_A_BLOCKS")==0){
    if (value>=PROPS[3].min_value && value<=PROPS[3].max_value){
       DEVICE_INFO[device_id].tune_params[2]=value;
       return PROPERTY_RETURN_OK;
    } else {
      return PROPERTY_RETURN_OUTSIDE_RANGE;
    }
  }
  if (strcmp(compare_buf,"GEN_A_TPB")==0){
    if (value>=PROPS[4].min_value && value<=PROPS[4].max_value){
       DEVICE_INFO[device_id].tune_params[3]=value;
       return PROPERTY_RETURN_OK;
    } else {
      return PROPERTY_RETURN_OUTSIDE_RANGE;
    }
  }
  if (strcmp(compare_buf,"GEN_B_TPB")==0){
    if (value>=PROPS[5].min_value && value<=PROPS[5].max_value){
       DEVICE_INFO[device_id].tune_params[4]=value;
       return PROPERTY_RETURN_OK;
    } else {
      return PROPERTY_RETURN_OUTSIDE_RANGE;
    }
  }
  if (strcmp(compare_buf,"TRIM_TPB")==0){
    if (value>=PROPS[6].min_value && value<=PROPS[6].max_value){
       DEVICE_INFO[device_id].tune_params[5]=value;
       return PROPERTY_RETURN_OK;
    } else {
      return PROPERTY_RETURN_OUTSIDE_RANGE;
    }
  }
  if (strcmp(compare_buf,"TAIL_TPB")==0){
    if (value>=PROPS[7].min_value && value<=PROPS[7].max_value){
       DEVICE_INFO[device_id].tune_params[6]=value;
       return PROPERTY_RETURN_OK;
    } else {
      return PROPERTY_RETURN_OUTSIDE_RANGE;
    }
  }
  if (strcmp(compare_buf,"RECOVER_BLOCKS")==0){
    if (value>=PROPS[8].min_value && value<=PROPS[8].max_value){
       DEVICE_INFO[device_id].tune_params[7]=value;
       return PROPERTY_RETURN_OK;
    } else {
      return PROPERTY_RETURN_OUTSIDE_RANGE;
    }
  }
  if (strcmp(compare_buf,"RECOVER_TPB")==0){
    if (value>=PROPS[9].min_value && value<=PROPS[9].max_value){
       DEVICE_INFO[device_id].tune_params[8]=value;
       return PROPERTY_RETURN_OK;
    } else {
      return PROPERTY_RETURN_OUTSIDE_RANGE;
    }
  }
  return PROPERTY_RETURN_NOT_FOUND;
}

extern "C" int cuckoo_get_parameter(char *param_name,
                                     int param_name_len,
                                     int device_id,
                                     int* value){
  if (param_name_len > MAX_PROPERTY_NAME_LENGTH) return PROPERTY_RETURN_TOO_LONG;
  if (device_id > NUM_DEVICES-1) return PROPERTY_RETURN_INVALID_DEVICE;
  char compare_buf[MAX_PROPERTY_NAME_LENGTH];
  snprintf(compare_buf,param_name_len+1,"%s", param_name);
  if (strcmp(compare_buf,"USE_DEVICE")==0){
       *value = DEVICE_INFO[device_id].use_device_param;
       return PROPERTY_RETURN_OK;
  }
  if (strcmp(compare_buf,"EXPAND")==0){
       *value = DEVICE_INFO[device_id].tune_params[0];
       return PROPERTY_RETURN_OK;
  }
  if (strcmp(compare_buf,"N_TRIMS")==0){
       *value = DEVICE_INFO[device_id].tune_params[1];
       return PROPERTY_RETURN_OK;
  }
  if (strcmp(compare_buf,"GEN_A_BLOCKS")==0){
       *value = DEVICE_INFO[device_id].tune_params[2];
       return PROPERTY_RETURN_OK;
  }
  if (strcmp(compare_buf,"GEN_A_TPB")==0){
       *value = DEVICE_INFO[device_id].tune_params[3];
       return PROPERTY_RETURN_OK;
  }
  if (strcmp(compare_buf,"GEN_B_TPB")==0){
       *value = DEVICE_INFO[device_id].tune_params[4];
       return PROPERTY_RETURN_OK;
  }
  if (strcmp(compare_buf,"TRIM_TPB")==0){
       *value = DEVICE_INFO[device_id].tune_params[5];
       return PROPERTY_RETURN_OK;
  }
  if (strcmp(compare_buf,"TAIL_TPB")==0){
       *value = DEVICE_INFO[device_id].tune_params[6];
       return PROPERTY_RETURN_OK;
  }
  if (strcmp(compare_buf,"RECOVER_BLOCKS")==0){
       *value = DEVICE_INFO[device_id].tune_params[7];
       return PROPERTY_RETURN_OK;
  }
  if (strcmp(compare_buf,"RECOVER_TPB")==0){
       *value = DEVICE_INFO[device_id].tune_params[8];
       return PROPERTY_RETURN_OK;
  }
  return PROPERTY_RETURN_NOT_FOUND;
}

bool cuckoo_internal_ready_for_data(){
  //just return okay if a device is flagged as free
  for (int i=0;i<NUM_DEVICES;i++){
    if (!DEVICE_INFO[i].is_busy && !DEVICE_INFO[i].threw_error && DEVICE_INFO[i].use_device_param){
       return true;
    }
  }
  return false;
}

struct InternalWorkerArgs {
  unsigned int id;
  unsigned int length;
  char data[MAX_DATA_LENGTH];
  unsigned char nonce[8];
  u32 device_id;
};

void update_stats(u32 device_id, u64 start_time) {
  pthread_mutex_lock (&device_info_mutex);
  DEVICE_INFO[device_id].last_start_time=start_time;
  DEVICE_INFO[device_id].last_end_time=timestamp();
  DEVICE_INFO[device_id].last_solution_time=DEVICE_INFO[device_id].last_end_time-
  DEVICE_INFO[device_id].last_start_time; 
  DEVICE_INFO[device_id].is_busy=false;
  DEVICE_INFO[device_id].iterations_completed++;
  pthread_mutex_unlock(&device_info_mutex);
}

void mark_device_error(u32 device_id) {
  pthread_mutex_lock (&device_info_mutex);
  DEVICE_INFO[device_id].threw_error = true;
  pthread_mutex_unlock(&device_info_mutex);
}

void *process_internal_worker (void *vp) {
  InternalWorkerArgs* args = (InternalWorkerArgs*) vp;

  //this should set the device for this thread
  cudaSetDevice(args->device_id);

  u32 response[PROOFSIZE];
  u64 start_time=timestamp();

  int return_val=cuckoo_call((char*) args->data, args->length, response);
  update_stats(args->device_id, start_time);

  if (return_val==1){
    QueueOutput output;
    memcpy(output.result_nonces, response, sizeof(output.result_nonces));
    memcpy(output.nonce, args->nonce, sizeof(output.nonce));
    //std::cout<<"Adding to queue "<<output.nonce<<std::endl;
    output.id = args->id;
    OUTPUT_QUEUE.enqueue(output);
  }
  delete(args);
  internal_processing_finished=true;
}

int cuckoo_internal_process_data(unsigned int id, unsigned char* data, int data_length, unsigned char* nonce){
    //Not entirely sure... this should select a free device, then send it to the next available
    InternalWorkerArgs* args=new InternalWorkerArgs();
    args->id = id;
    args->length = data_length;
    memcpy(args->data, data, data_length);
    memcpy(args->nonce, nonce, sizeof(args->nonce));
    u32 device_id=next_free_device_id();
    args->device_id=device_id;
    pthread_mutex_lock(&device_info_mutex);
    DEVICE_INFO[device_id].is_busy=true;
    DEVICE_INFO[device_id].device_id=device_id;
    pthread_mutex_unlock(&device_info_mutex);
    pthread_t internal_worker_thread;
    is_working=true;
    if (should_quit) return 1;
    internal_processing_finished=false;
    if (!pthread_create(&internal_worker_thread, NULL, process_internal_worker, args)){
        if (pthread_detach(internal_worker_thread)){
            return 1;
        }
    }
    return 0;
}
#endif
