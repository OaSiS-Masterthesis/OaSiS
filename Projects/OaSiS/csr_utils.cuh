#ifndef CSR_UTILS_CUH
#define CSR_UTILS_CUH

#include "iq.cuh"

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

class MatrixOperations{
public:
	using streamIdx		 = Cuda::StreamIndex;
	using eventIdx		 = Cuda::EventIndex;
		
	gko::array<int> temporary_rows;
	gko::array<int> temporary_columns;
	gko::array<float> temporary_values;
	
	void initialize(std::shared_ptr<gko::Executor>& ginkgo_executor){
		temporary_rows = gko::array<int>(ginkgo_executor, 32 * config::G_BLOCKVOLUME + 1);
		temporary_columns = gko::array<int>(ginkgo_executor, 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		temporary_values = gko::array<float>(ginkgo_executor,  32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
	}
	
	//Sum up count values from in and store them in out
	template<typename CudaContext>
	void exclusive_scan(int count, int const* const in, int* out, CudaContext& cu_dev) {
		std::cout << "Thrust call 0 Start. Count: " << count << std::endl;
		auto policy = thrust::cuda::par.on(static_cast<cudaStream_t>(cu_dev.stream_compute()));
		thrust::exclusive_scan(policy, get_device_ptr(in), get_device_ptr(in) + count, get_device_ptr(out));
		std::cout << "Thrust call 0 End." << std::endl;
		/*
		std::size_t temp_storage_bytes = 0;
		auto plus_op				   = [] __device__(const int& a, const int& b) {
			  return a + b;
		};
		check_cuda_errors(cub::DeviceScan::ExclusiveScan(nullptr, temp_storage_bytes, in, out, plus_op, 0, count, cu_dev.stream_compute()));
		void* d_tmp = tmps[cu_dev.get_dev_id()].d_tmp;
		check_cuda_errors(cub::DeviceScan::ExclusiveScan(d_tmp, temp_storage_bytes, in, out, plus_op, 0, count, cu_dev.stream_compute()));
		*/
	}
	
	//FIXME: Untested
	void matrix_transpose(std::shared_ptr<gko::Executor>& ginkgo_executor, const size_t num_blocks, const size_t num_columns, std::shared_ptr<gko::matrix::Csr<float, int>>& a, std::shared_ptr<gko::matrix::Csr<float, int>>& b, Cuda::CudaContext& cu_dev){	
		temporary_rows.resize_and_reset(num_columns + 1);
		
		temporary_rows.fill(0);
		
		ginkgo_executor->synchronize();
		
		//Copy last rows value
		const int number_of_nonzeros_a = a->get_num_stored_elements();
		cudaMemcpyAsync(a->get_row_ptrs() + (num_blocks * config::G_BLOCKVOLUME), &number_of_nonzeros_a, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
		
		//Resize matrix
		temporary_columns.resize_and_reset(number_of_nonzeros_a);
		temporary_values.resize_and_reset(number_of_nonzeros_a);
		
		temporary_columns.fill(0);
		temporary_values.fill(0.0f);
		
		if(number_of_nonzeros_a > 0){
			//Transpose
			cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_transpose<iq::NUM_ROWS_PER_BLOCK, 1>, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), temporary_rows.get_data(), temporary_columns.get_data(), temporary_values.get_data());
			
			//Scan rows
			exclusive_scan(num_columns + 1, temporary_rows.get_data(), temporary_rows.get_data(), cu_dev);
		}
		
		//Copy to output
		b->copy_from(std::move(gko::share(gko::matrix::Csr<float, int>::create_const(
			  ginkgo_executor
			, gko::dim<2>(num_columns, num_blocks * config::G_BLOCKVOLUME)
			, temporary_values.as_const_view()
			, temporary_columns.as_const_view()
			, temporary_rows.as_const_view()
		))));
		
		//Sort columns and values
		b->sort_by_column_index();
		
		ginkgo_executor->synchronize();
	}
	
	//Calculates C = A * B (Gustavson == true) or C = A * B^T
	template<bool Gustavson = false>
	void matrix_matrix_multiplication(std::shared_ptr<gko::Executor>& ginkgo_executor, const size_t num_blocks, const size_t num_columns, std::shared_ptr<gko::matrix::Csr<float, int>>& a, std::shared_ptr<gko::matrix::Csr<float, int>>& b, std::shared_ptr<gko::matrix::Csr<float, int>>& c, Cuda::CudaContext& cu_dev){					
		temporary_rows.resize_and_reset(num_blocks * config::G_BLOCKVOLUME + 1);
		
		temporary_rows.fill(0);
		
		ginkgo_executor->synchronize();
		
		//Copy last rows value
		const int number_of_nonzeros_a = a->get_num_stored_elements();
		const int number_of_nonzeros_b = b->get_num_stored_elements();
		cudaMemcpyAsync(a->get_row_ptrs() + num_blocks * config::G_BLOCKVOLUME, &number_of_nonzeros_a, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
		cudaMemcpyAsync(b->get_row_ptrs() + num_blocks * config::G_BLOCKVOLUME, &number_of_nonzeros_b, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
		
		if(Gustavson){
			//Resize temporary memory
			temporary_columns.resize_and_reset(num_blocks * num_columns);
			
			ginkgo_executor->synchronize();
		}
		
		//Calculate amount of memory
		if(Gustavson){
			cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication_gustavson_calculate_rows<iq::NUM_ROWS_PER_BLOCK, 1>, num_columns, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), temporary_rows.get_data(), temporary_columns.get_data());
		}else{
			cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication<iq::NUM_ROWS_PER_BLOCK, 1, true>, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), temporary_rows.get_data(), nullptr, nullptr);
		}
		
		//Scan rows
		exclusive_scan(num_blocks * config::G_BLOCKVOLUME + 1, temporary_rows.get_data(), temporary_rows.get_data(), cu_dev);
		
		//Resize matrix
		int number_of_nonzeros_host;
		
		cudaMemcpyAsync(&number_of_nonzeros_host, temporary_rows.get_data() + (num_blocks * config::G_BLOCKVOLUME), sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
		
		cu_dev.syncStream<streamIdx::COMPUTE>();
		
		temporary_columns.resize_and_reset(number_of_nonzeros_host);
		temporary_values.resize_and_reset(number_of_nonzeros_host);
		
		temporary_columns.fill(0);
		temporary_values.fill(0.0f);
		
		ginkgo_executor->synchronize();
		
		//Perform matrix multiplication
		if(number_of_nonzeros_host > 0){
			if(Gustavson){
				cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication_gustavson<iq::NUM_ROWS_PER_BLOCK, 1>, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), temporary_rows.get_data(), temporary_columns.get_data(), temporary_values.get_data());
			}else{
				cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication<iq::NUM_ROWS_PER_BLOCK, 1, false>, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), temporary_rows.get_data(), temporary_columns.get_data(), temporary_values.get_data());
			}
		}
		
		//Copy to output
		c->copy_from(std::move(gko::share(gko::matrix::Csr<float, int>::create_const(
			  ginkgo_executor
			, gko::dim<2>(num_blocks * config::G_BLOCKVOLUME, num_columns)
			, temporary_values.as_const_view()
			, temporary_columns.as_const_view()
			, temporary_rows.as_const_view()
		))));
		
		ginkgo_executor->synchronize();
		//NOTE: Columns already sorted
	}
	
	//Calculates C = A * D * B (Gustavson == true) or C = A * D * B^T
	template<bool Gustavson = false>
	void matrix_matrix_multiplication_with_diagonal(std::shared_ptr<gko::Executor>& ginkgo_executor, const size_t num_blocks, const size_t num_columns, std::shared_ptr<gko::matrix::Csr<float, int>>& a, const std::shared_ptr<const gko::matrix::Diagonal<float>>& d, std::shared_ptr<gko::matrix::Csr<float, int>>& b, std::shared_ptr<gko::matrix::Csr<float, int>>& c, Cuda::CudaContext& cu_dev){					
		temporary_rows.resize_and_reset(num_blocks * config::G_BLOCKVOLUME + 1);
		
		temporary_rows.fill(0);
		
		ginkgo_executor->synchronize();
		
		//Copy last rows value
		const int number_of_nonzeros_a = a->get_num_stored_elements();
		const int number_of_nonzeros_b = b->get_num_stored_elements();
		cudaMemcpyAsync(a->get_row_ptrs() + num_blocks * config::G_BLOCKVOLUME, &number_of_nonzeros_a, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
		cudaMemcpyAsync(b->get_row_ptrs() + num_blocks * config::G_BLOCKVOLUME, &number_of_nonzeros_b, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
		
		if(Gustavson){
			//Resize temporary memory
			temporary_columns.resize_and_reset(num_blocks * num_columns);
			
			ginkgo_executor->synchronize();
		}
		
		//Calculate amount of memory
		if(Gustavson){
			cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication_with_diagonal_gustavson_calculate_rows<iq::NUM_ROWS_PER_BLOCK, 1>, num_columns, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), d->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), temporary_rows.get_data(), temporary_columns.get_data());
		}else{
			cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication_with_diagonal<iq::NUM_ROWS_PER_BLOCK, 1, true>, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), d->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), temporary_rows.get_data(), nullptr, nullptr);
		}
		
		//Scan rows
		exclusive_scan(num_blocks * config::G_BLOCKVOLUME + 1, temporary_rows.get_data(), temporary_rows.get_data(), cu_dev);
		
		//Resize matrix
		int number_of_nonzeros_host;
		
		cudaMemcpyAsync(&number_of_nonzeros_host, temporary_rows.get_data() + (num_blocks * config::G_BLOCKVOLUME), sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
		
		cu_dev.syncStream<streamIdx::COMPUTE>();
		
		temporary_columns.resize_and_reset(number_of_nonzeros_host);
		temporary_values.resize_and_reset(number_of_nonzeros_host);
		
		temporary_columns.fill(0);
		temporary_values.fill(0.0f);
		
		ginkgo_executor->synchronize();
		
		//Perform matrix multiplication
		if(number_of_nonzeros_host > 0){
			if(Gustavson){
				cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication_with_diagonal_gustavson<iq::NUM_ROWS_PER_BLOCK, 1>, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), d->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), temporary_rows.get_data(), temporary_columns.get_data(), temporary_values.get_data());
			}else{
				cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication_with_diagonal<iq::NUM_ROWS_PER_BLOCK, 1, false>, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), d->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), temporary_rows.get_data(), temporary_columns.get_data(), temporary_values.get_data());
			}
		}
		
		//Copy to output
		c->copy_from(std::move(gko::share(gko::matrix::Csr<float, int>::create_const(
			  ginkgo_executor
			, gko::dim<2>(num_blocks * config::G_BLOCKVOLUME, num_columns)
			, temporary_values.as_const_view()
			, temporary_columns.as_const_view()
			, temporary_rows.as_const_view()
		))));
		
		ginkgo_executor->synchronize();
		//NOTE: Columns already sorted
	}
	
	//Calculates C = A * A^T
	void matrix_matrix_multiplication_a_at(std::shared_ptr<gko::Executor>& ginkgo_executor, const size_t num_blocks, const size_t num_columns, std::shared_ptr<gko::matrix::Csr<float, int>>& a, std::shared_ptr<gko::matrix::Csr<float, int>>& a_transposed, std::shared_ptr<gko::matrix::Csr<float, int>>& c, Cuda::CudaContext& cu_dev){					
		matrix_matrix_multiplication<true>(ginkgo_executor, num_blocks, num_columns, a, a_transposed, c, cu_dev);
		
		if(c->get_num_stored_elements() > 0){
			//Mirror
			cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_mirror<iq::NUM_ROWS_PER_BLOCK, 1>, c->get_const_row_ptrs(), c->get_const_col_idxs(), c->get_values());
		}
	}
	
	//Calculates C = A * D * A^T
	void matrix_matrix_multiplication_a_at_with_diagonal(std::shared_ptr<gko::Executor>& ginkgo_executor, const size_t num_blocks, const size_t num_columns, std::shared_ptr<gko::matrix::Csr<float, int>>& a, const std::shared_ptr<const gko::matrix::Diagonal<float>>& d, std::shared_ptr<gko::matrix::Csr<float, int>>& a_transposed, std::shared_ptr<gko::matrix::Csr<float, int>>& c, Cuda::CudaContext& cu_dev){					
		matrix_matrix_multiplication_with_diagonal<true>(ginkgo_executor, num_blocks, num_columns, a, d, a_transposed, c, cu_dev);
		
		if(c->get_num_stored_elements() > 0){
			//Mirror
			cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_mirror<iq::NUM_ROWS_PER_BLOCK, 1>, c->get_const_row_ptrs(), c->get_const_col_idxs(), c->get_values());
		}
	}
};

//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers)
}// namespace mn

#endif