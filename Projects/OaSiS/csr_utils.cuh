#ifndef CSR_UTILS_CUH
#define CSR_UTILS_CUH

namespace mn {
using namespace placeholder;//NOLINT(google-build-using-namespace) Allow placeholders to be included generally for simplification

//TODO: Make magic numbers to constants where suitable
//TODO: Ensure call dimensions and such are small enough to allow narrowing conversations. Or directly use unsigned where possible
//TODO: Maybe use names instead of formula signs for better understanding
//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers) CUDA does not yet support std::span; Common names for physical formulas; Cannot declare __global__ functions inline

//Sliced
template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t get_thread_index(const size_t index){
	return index % BLOCK_SIZE;
}

template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t get_thread_offset(const size_t index){
	return index / BLOCK_SIZE;
}

template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t get_thread_count(const size_t thread_id, const size_t global_count){
	return (global_count / BLOCK_SIZE) + (global_count % BLOCK_SIZE > thread_id ? 1 : 0);
}

template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t get_global_index(const size_t thread_id, const size_t offset){
	return offset * BLOCK_SIZE + thread_id;
}

//NOTE: This may also verify symmetric with using one matrix for both original and transposed
template<size_t NumRowsPerBlock, size_t NumDimensionsPerRow, size_t NumDimensionsPerRowTransposed>
__global__ void csr_verify_transposed(const int* iq_rows, const int* iq_columns, const float* iq_values, const int* iq_rows_transposed, const int* iq_columns_transposed, const float* iq_values_transposed) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	//const auto blockid			   = partition.active_keys[blockIdx.x];
	
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		for(size_t dimension = 0; dimension < NumDimensionsPerRow; ++dimension){
			for(size_t column_index = iq_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension]; column_index < iq_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension + 1]; ++column_index){
				const int column = iq_columns[column_index];
				//Find transposed
				int found_column_index_transposed = -1;
				for(size_t column_index_transposed = iq_rows_transposed[column]; column_index_transposed < iq_rows_transposed[column + 1]; ++column_index_transposed){
					const int column_transposed = iq_columns_transposed[column_index_transposed];
					if(column_transposed == (NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension)){
						found_column_index_transposed = column_index_transposed;
						break;
					}
				}
				
				if(found_column_index_transposed != -1){
					if(iq_values[column_index] != iq_values_transposed[found_column_index_transposed]){
						printf("ERROR - Original and Transposed do not match in (%d %d)\n", static_cast<int>(NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension), column);
					}
				}else{
					if(iq_values[column_index] != 0.0f){
						printf("ERROR - Original and Transposed do not match in (%d %d)\n", static_cast<int>(NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension), column);
					}
				}
			}
		}
	}
	
	//Check transposed
	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		for(size_t dimension = 0; dimension < NumDimensionsPerRowTransposed; ++dimension){
			for(size_t column_index = iq_rows_transposed[NumDimensionsPerRowTransposed * base_row + NumDimensionsPerRowTransposed * row + dimension]; column_index < iq_rows_transposed[NumDimensionsPerRowTransposed * base_row + NumDimensionsPerRowTransposed * row + dimension + 1]; ++column_index){
				const int column = iq_columns_transposed[column_index];
				//Find transposed
				int found_column_index_transposed = -1;
				for(size_t column_index_transposed = iq_rows[column]; column_index_transposed < iq_rows[column + 1]; ++column_index_transposed){
					const int column_transposed = iq_columns[column_index_transposed];
					if(column_transposed == (NumDimensionsPerRowTransposed * base_row + NumDimensionsPerRowTransposed * row + dimension)){
						found_column_index_transposed = column_index_transposed;
						break;
					}
				}
				
				if(found_column_index_transposed != -1){
					if(iq_values_transposed[column_index] != iq_values[found_column_index_transposed]){
						printf("ERROR - Transposed and Original do not match in (%d %d)\n", static_cast<int>(NumDimensionsPerRowTransposed * base_row + NumDimensionsPerRowTransposed * row + dimension), column);
					}
				}else{
					if(iq_values_transposed[column_index] != 0.0f){
						printf("ERROR - Transposed and Original do not match in (%d %d)\n", static_cast<int>(NumDimensionsPerRowTransposed * base_row + NumDimensionsPerRowTransposed * row + dimension), column);
					}
				}
			}
		}
	}
}

//FIXME: Untested
template<size_t NumRowsPerBlock, size_t NumDimensionsPerRow>
__global__ void csr_transpose(const int* iq_rows, const int* iq_columns, const float* iq_values, int* iq_rows_transposed, int* iq_columns_transposed, float* iq_values_transposed) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	//const auto blockid			   = partition.active_keys[blockIdx.x];
	
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		for(size_t dimension = 0; dimension < NumDimensionsPerRow; ++dimension){
			for(size_t column_index = iq_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension]; column_index < iq_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension + 1]; ++column_index){
				const int column = iq_columns[column_index];
				
				//Write row to all columns in row
				const int column_index_transposed = atomicAdd(&(iq_rows_transposed[column]), 1);
				iq_columns_transposed[column_index_transposed] = NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension;
				iq_values_transposed[column_index_transposed] = iq_values[column_index];
			}
		}
	}
}

//A * B^T
template<size_t NumRowsPerBlock, size_t NumDimensionsPerRow, bool WriteRows>
__global__ void csr_matrix_matrix_multiplication(const int* a_rows, const int* a_columns, const float* a_values, const int* b_rows, const int* b_columns, const float* b_values, int* c_rows, int* c_columns, float* c_values) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	//const auto blockid			   = partition.active_keys[blockIdx.x];
	
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	//For all rows in A
	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		for(size_t dimension_row = 0; dimension_row < NumDimensionsPerRow; ++dimension_row){
			
			//Index fetched once, then incremented as we iterate over all columns
			int column_index_c = c_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row];
			
			//For all columns in B^T == rows in B 
			for(size_t column = 0; column < gridDim.x * NumRowsPerBlock; ++column){
				for(size_t dimension_column = 0; dimension_column < NumDimensionsPerRow; ++dimension_column){
					
					//Index fetched again for each column for correct dot product
					int column_index_a = a_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row];
					int column_index_b = b_rows[NumDimensionsPerRow * column + dimension_column];
					
					//sum = dot(a_row, b_column)
					float sum = 0.0f;
					while(
						   (column_index_a < a_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row + 1])
						&& (column_index_b < b_rows[NumDimensionsPerRow * column + dimension_column + 1])
					){
						const int column_a = a_columns[column_index_a];
						const int column_b = b_columns[column_index_b];
						
						if(column_a < column_b){
							column_index_a++;
						}else if(column_b < column_a){
							column_index_b++;
						}else{//column_a == column_b
							const float value_a = a_values[column_index_a];
							const float value_b = b_values[column_index_b];
							
							sum += value_a * value_b;
							
							column_index_a++;
							column_index_b++;
						}
					}
					
					if(sum != 0.0f){
						if(WriteRows){
							c_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row] += 1;
						}else{
							c_columns[column_index_c] = NumDimensionsPerRow * column + dimension_column;
							c_values[column_index_c] = sum;
							
							column_index_c++;
						}
					}
				}
			}
		}
	}
}


template<size_t NumRowsPerBlock, size_t NumDimensionsPerRow>
__global__ void csr_matrix_matrix_multiplication_gustavson_calculate_rows(const size_t num_columns, const int* a_rows, const int* a_columns, const float* a_values, const int* b_rows, const int* b_columns, const float* b_values, int* c_rows, int* tmp_memory) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	//const auto blockid			   = partition.active_keys[blockIdx.x];
	
	int* tmp_memory_of_block = tmp_memory + blockIdx.x * num_columns;
	
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	//For all rows in A
	for(size_t row = 0; row < NumRowsPerBlock; ++row){
		for(size_t dimension_row = 0; dimension_row < NumDimensionsPerRow; ++dimension_row){
			
			//Clear temporary memory
			for(size_t temp_index = threadIdx.x; temp_index < num_columns; temp_index += blockDim.x){
				tmp_memory_of_block[temp_index] = 0;
			}
			
			__syncthreads();
			
			//For all columns in row_a
			for(int column_index_a = a_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row] + static_cast<int>(threadIdx.x); column_index_a < a_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row + 1]; column_index_a += static_cast<int>(blockDim.x)){
				const int column_a = a_columns[column_index_a];
				const float value_a = a_values[column_index_a];
				
				if(value_a != 0.0f){
					//For all rows in column_a
					for(int column_index_b = b_rows[column_a]; column_index_b < b_rows[column_a + 1]; ++column_index_b){
						const int column_b = b_columns[column_index_b];
						const float value_b = b_values[column_index_b];
						
						if(value_b != 0.0f){
							atomicAdd(tmp_memory_of_block + column_b, 1);
						}
					}
				}
			}
			
			__syncthreads();
			
			//Sum up
			for(size_t temp_index = threadIdx.x; temp_index < num_columns; temp_index += blockDim.x){
				if(tmp_memory_of_block[temp_index] > 0){
					atomicAdd(&(c_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row]), 1);
				}
			}
			
			__syncthreads();
		}
	}
}

//A * B
//Using Gustavson
template<size_t NumRowsPerBlock, size_t NumDimensionsPerRow>
__global__ void csr_matrix_matrix_multiplication_gustavson(const int* a_rows, const int* a_columns, const float* a_values, const int* b_rows, const int* b_columns, const float* b_values, const int* c_rows, int* c_columns, float* c_values) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	//const auto blockid			   = partition.active_keys[blockIdx.x];
	
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	//For all rows in A
	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		for(size_t dimension_row = 0; dimension_row < NumDimensionsPerRow; ++dimension_row){
			size_t column_count_row = 0;
			
			const int column_index_c = c_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row];
			const int num_columns = c_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row + 1] - column_index_c;
			int* columns_row = c_columns + column_index_c;
			float* values_row = c_values + column_index_c;
			
			thrust::fill(thrust::seq, columns_row, columns_row + num_columns, std::numeric_limits<int>::max());
			thrust::fill(thrust::seq, values_row, values_row + num_columns, 0.0f);
			
			//For all columns in row_a
			for(int column_index_a = a_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row]; column_index_a < a_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row + 1]; ++column_index_a){
				const int column_a = a_columns[column_index_a];
				const float value_a = a_values[column_index_a];
				
				if(value_a != 0.0f){
					//For all rows in column_a
					for(int column_index_b = b_rows[column_a]; column_index_b < b_rows[column_a + 1]; ++column_index_b){
						const int column_b = b_columns[column_index_b];
						const float value_b = b_values[column_index_b];
						
						if(value_b != 0.0f){
							//Binary search for column
							size_t l = 0;
							size_t r = column_count_row;
							
							size_t index = (l + r)/2;
							while(l < r && columns_row[index] != column_b){
								if(column_b < columns_row[index]){
									r = index - std::min(index, static_cast<size_t>(1));//Clamp result to 0
								}else{//column_b > columns_row[index]
									l = index + 1;
								}
								
								index = (l + r)/2;
							}
							
							if(index >= num_columns){
								printf("ERROR: Index out of bounds: %d\n", static_cast<int>(index));
							}
							
							//If column does not exist, insert it
							if(columns_row[index] != column_b){
								//Might be one of
								if(columns_row[index] < column_b){
									index++;
								}
								
								//Move values
								for(size_t copy_index = column_count_row; copy_index > index; --copy_index){
									columns_row[copy_index] = columns_row[copy_index - 1];
									values_row[copy_index] = values_row[copy_index - 1];
								}
								//Increase size
								column_count_row++;
								
								//Store coluimn
								columns_row[index] = column_b;
							}
							
							//Store value
							values_row[index] += value_a * value_b;//c_row(column_b) += a_row_column_a * b_column_a_column_b
						}
					}
				}
			}
		}
	}
}

//A * D * B^T
template<size_t NumRowsPerBlock, size_t NumDimensionsPerRow, bool WriteRows>
__global__ void csr_matrix_matrix_multiplication_with_diagonal(const int* a_rows, const int* a_columns, const float* a_values, const float* d_values, const int* b_rows, const int* b_columns, const float* b_values, int* c_rows, int* c_columns, float* c_values) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	//const auto blockid			   = partition.active_keys[blockIdx.x];
	
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	//For all rows in A
	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		for(size_t dimension_row = 0; dimension_row < NumDimensionsPerRow; ++dimension_row){
			
			//Index fetched once, then incremented as we iterate over all columns
			int column_index_c = c_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row];
			
			//For all columns in B^T == rows in B 
			for(size_t column = 0; column < gridDim.x * NumRowsPerBlock; ++column){
				for(size_t dimension_column = 0; dimension_column < NumDimensionsPerRow; ++dimension_column){
					
					//Index fetched again for each column for correct dot product
					int column_index_a = a_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row];
					int column_index_b = b_rows[NumDimensionsPerRow * column + dimension_column];
					
					//sum = dot(a_row, b_column)
					float sum = 0.0f;
					while(
						   (column_index_a < a_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row + 1])
						&& (column_index_b < b_rows[NumDimensionsPerRow * column + dimension_column + 1])
					){
						const int column_a = a_columns[column_index_a];
						const int column_b = b_columns[column_index_b];
						
						if(column_a < column_b){
							column_index_a++;
						}else if(column_b < column_a){
							column_index_b++;
						}else{//column_a == column_b
							const float value_a = a_values[column_index_a];
							const float value_b = b_values[column_index_b];
							const float value_d = d_values[column_a];
							
							sum += (value_a * value_b) * value_d;
							
							column_index_a++;
							column_index_b++;
						}
					}
					
					if(sum != 0.0f){
						if(WriteRows){
							c_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row] += 1;
						}else{
							c_columns[column_index_c] = NumDimensionsPerRow * column + dimension_column;
							c_values[column_index_c] = sum;
							
							column_index_c++;
						}
					}
				}
			}
		}
	}
}


template<size_t NumRowsPerBlock, size_t NumDimensionsPerRow>
__global__ void csr_matrix_matrix_multiplication_with_diagonal_gustavson_calculate_rows(const size_t num_columns, const int* a_rows, const int* a_columns, const float* a_values, const float* d_values, const int* b_rows, const int* b_columns, const float* b_values, int* c_rows, int* tmp_memory) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	//const auto blockid			   = partition.active_keys[blockIdx.x];
	
	int* tmp_memory_of_block = tmp_memory + blockIdx.x * num_columns;
	
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	//For all rows in A
	for(size_t row = 0; row < NumRowsPerBlock; ++row){
		for(size_t dimension_row = 0; dimension_row < NumDimensionsPerRow; ++dimension_row){
			
			//Clear temporary memory
			for(size_t temp_index = threadIdx.x; temp_index < num_columns; temp_index += blockDim.x){
				tmp_memory_of_block[temp_index] = 0;
			}
			
			__syncthreads();
			
			//For all columns in row_a
			for(int column_index_a = a_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row] + static_cast<int>(threadIdx.x); column_index_a < a_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row + 1]; column_index_a += static_cast<int>(blockDim.x)){
				const int column_a = a_columns[column_index_a];
				const float value_a = a_values[column_index_a];
				
				const float value_d = d_values[column_a];
				
				if(value_a != 0.0f && value_d != 0.0f){
					//For all rows in column_a
					for(int column_index_b = b_rows[column_a]; column_index_b < b_rows[column_a + 1]; ++column_index_b){
						const int column_b = b_columns[column_index_b];
						const float value_b = b_values[column_index_b];
						
						if(value_b != 0.0f){
							if(column_b >= num_columns){
								printf("Num columns wrong!");
							}
							atomicAdd(tmp_memory_of_block + column_b, 1);
						}
					}
				}
			}
			
			__syncthreads();
			
			//Sum up
			for(size_t temp_index = threadIdx.x; temp_index < num_columns; temp_index += blockDim.x){
				if(tmp_memory_of_block[temp_index] > 0){
					atomicAdd(&(c_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row]), 1);
				}
			}
			
			__syncthreads();
		}
	}
}

//A * D * B
//Using Gustavson
template<size_t NumRowsPerBlock, size_t NumDimensionsPerRow>
__global__ void csr_matrix_matrix_multiplication_with_diagonal_gustavson(const int* a_rows, const int* a_columns, const float* a_values, const float* d_values, const int* b_rows, const int* b_columns, const float* b_values, const int* c_rows, int* c_columns, float* c_values) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	//const auto blockid			   = partition.active_keys[blockIdx.x];
	
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	//For all rows in A
	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		for(size_t dimension_row = 0; dimension_row < NumDimensionsPerRow; ++dimension_row){
			size_t column_count_row = 0;
			
			const int column_index_c = c_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row];
			const int num_columns = c_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row + 1] - column_index_c;
			int* columns_row = c_columns + column_index_c;
			float* values_row = c_values + column_index_c;
			
			thrust::fill(thrust::seq, columns_row, columns_row + num_columns, std::numeric_limits<int>::max());
			thrust::fill(thrust::seq, values_row, values_row + num_columns, 0.0f);
			
			//For all columns in row_a
			for(int column_index_a = a_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row]; column_index_a < a_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row + 1]; ++column_index_a){
				const int column_a = a_columns[column_index_a];
				const float value_a = a_values[column_index_a];
				
				const float value_d = d_values[column_a];
				
				if(value_a != 0.0f && value_d != 0.0f){
					//For all rows in column_a
					for(int column_index_b = b_rows[column_a]; column_index_b < b_rows[column_a + 1]; ++column_index_b){
						const int column_b = b_columns[column_index_b];
						const float value_b = b_values[column_index_b];
						
						if(value_b != 0.0f){
							//Binary search for column
							size_t l = 0;
							size_t r = column_count_row;
							
							size_t index = (l + r)/2;
							while(l < r && columns_row[index] != column_b){
								if(column_b < columns_row[index]){
									r = index - std::min(index, static_cast<size_t>(1));//Clamp result to 0
								}else{//column_b > columns_row[index]
									l = index + 1;
								}
								
								index = (l + r)/2;
							}
							
							if(index >= num_columns){
								printf("ERROR: Index out of bounds: %d\n", static_cast<int>(index));
							}
							
							//If column does not exist, insert it
							if(columns_row[index] != column_b){
								//Might be one of
								if(columns_row[index] < column_b){
									index++;
								}
								
								//Move values
								for(size_t copy_index = column_count_row; copy_index > index; --copy_index){
									columns_row[copy_index] = columns_row[copy_index - 1];
									values_row[copy_index] = values_row[copy_index - 1];
								}
								//Increase size
								column_count_row++;
								
								//Store coluimn
								columns_row[index] = column_b;
							}
							
							//Store value
							values_row[index] += (value_a * value_b) * value_d;//c_row(column_b) += a_row_column_a * d_column_a * b_column_a_column_b
						}
					}
				}
			}
		}
	}
}

//Copy upper triangular to lower triangular
template<size_t NumRowsPerBlock, size_t NumDimensionsPerRow>
__global__ void csr_mirror(const int* a_rows, const int* a_columns, float* a_values) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	//const auto blockid			   = partition.active_keys[blockIdx.x];
	
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	//For all rows in A
	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		for(size_t dimension_row = 0; dimension_row < NumDimensionsPerRow; ++dimension_row){
			
			//For all columns in row_a
			for(int column_index_a = a_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row]; column_index_a < a_rows[NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row + 1]; ++column_index_a){
				const int column_a = a_columns[column_index_a];
				
				if(column_a < (NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row)){
					for(int column_index_a_transposed = a_rows[column_a]; column_index_a_transposed < a_rows[column_a + 1]; ++column_index_a_transposed){
						const int column_a_transposed = a_columns[column_index_a_transposed];
						
						if(column_a_transposed == (NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension_row)){
							a_values[column_index_a] = a_values[column_index_a_transposed];
							break;
						}
					}
					
				}
			}
		}
	}
}



//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers)
}// namespace mn

#endif