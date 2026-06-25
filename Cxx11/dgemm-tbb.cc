///
/// Copyright (c) 2026, Intel Corporation
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions
/// are met:
///
/// * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
/// * Redistributions in binary form must reproduce the above
///       copyright notice, this list of conditions and the following
///       disclaimer in the documentation and/or other materials provided
///       with the distribution.
/// * Neither the name of Intel Corporation nor the names of its
///       contributors may be used to endorse or promote products
///       derived from this software without specific prior written
///       permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
/// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
/// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
/// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
/// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
/// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
/// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
/// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
/// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
/// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
/// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.

//////////////////////////////////////////////////////////////////////
///
/// NAME:    dgemm
///
/// PURPOSE: This program tests the efficiency with which a dense matrix
///          dense multiplication is carried out
///
/// USAGE:   The program takes as input the matrix order and
///          the number of times the matrix-matrix multiplication
///          is carried out
///
///          <progname> <# iterations> <matrix order>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than OpenMP or standard C functions, the following
///          functions are used in this program:
///
///          wtime()
///
/// HISTORY: Written by Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, December, 2017.
///          Parallelized with TBB (blocked_range2d) for this version.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_tbb.h"

int main(int argc, char * argv[])
{
  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  std::cout << "Parallel Research Kernels" << std::endl;
  std::cout << "C++11/TBB Dense matrix-matrix multiplication: C += A x B" << std::endl;

  int iterations;
  int order;
  try {
    if (argc < 3) {
      throw "Usage: <# iterations> <matrix order>";
    }

    iterations = std::atoi(argv[1]);
    if (iterations < 1) {
      throw "ERROR: iterations must be >= 1";
    }

    order = std::atoi(argv[2]);
    if (order <= 0) {
      throw "ERROR: Matrix Order must be greater than 0";
    } else if (order > prk::get_max_matrix_size()) {
      throw "ERROR: matrix dimension too large - overflow risk";
    }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  const char* envvar = std::getenv("TBB_NUM_THREADS");
  int num_threads = (envvar != NULL) ? std::atoi(envvar) : prk::get_num_cores();
  tbb::global_control c(tbb::global_control::max_allowed_parallelism, num_threads);

  std::cout << "Number of threads    = " << num_threads << std::endl;
  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Matrix order         = " << order << std::endl;
  std::cout << "TBB partitioner      = " << tbb_partitioner_name << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for matrices
  //////////////////////////////////////////////////////////////////////

  double dgemm_time{0};

  const size_t elems = static_cast<size_t>(order) * static_cast<size_t>(order);
  const size_t bytes = elems * sizeof(double);

#if TBB_HAS_NUMA_ALLOCATION
  auto deleter = [bytes](double * p) { tbb::deallocate_numa_interleaved(p, bytes); };
  using numa_ptr = std::unique_ptr<double, decltype(deleter)>;

  numa_ptr pA(static_cast<double*>(tbb::allocate_numa_interleaved(bytes)), deleter);
  numa_ptr pB(static_cast<double*>(tbb::allocate_numa_interleaved(bytes)), deleter);
  numa_ptr pC(static_cast<double*>(tbb::allocate_numa_interleaved(bytes)), deleter);

  if (!pA || !pB || !pC) {
    std::cout << "ERROR: NUMA interleaved allocation failed" << std::endl;
    return 1;
  }
#else
  auto pA = std::make_unique<double[]>(elems);
  auto pB = std::make_unique<double[]>(elems);
  auto pC = std::make_unique<double[]>(elems);
#endif

  double * const A = pA.get();
  double * const B = pB.get();
  double * const C = pC.get();

  for (int i = 0; i < order; ++i) {
    for (int j = 0; j < order; ++j) {
      A[i * order + j] = i;
      B[i * order + j] = i;
      C[i * order + j] = 0.0;
    }
  }

  for (int iter = 0; iter <= iterations; iter++) {
    if (iter == 1)
      dgemm_time = prk::wtime();

    tbb::parallel_for(
        tbb::blocked_range2d<int>(0, order, 64, 0, order, 32),
        [&](const tbb::blocked_range2d<int> &r) {
          const int ibegin = r.rows().begin();
          const int iend = r.rows().end();
          const int jbegin = r.cols().begin();
          const int jend = r.cols().end();
          for (int i = ibegin; i < iend; ++i) {
            for (int k = 0; k < order; ++k) {
              const double aik = A[i * order + k];
              PRAGMA_SIMD
              for (int j = jbegin; j < jend; ++j) {
                C[i * order + j] += aik * B[k * order + j];
              }
            }
          }
        },
        tbb_partitioner);
  }
  dgemm_time = prk::wtime() - dgemm_time;

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  const auto forder = static_cast<double>(order);
  const auto reference = 0.25 * prk::pow(forder, 3) * prk::pow(forder - 1.0, 2) * (iterations + 1);
  const auto checksum = prk::reduce(C, C + elems, 0.0);

  const auto epsilon = 1.0e-8;
  const auto residuum = prk::abs(checksum - reference) / reference;
  if (residuum < epsilon) {
#if VERBOSE
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
#endif
    std::cout << "Solution validates" << std::endl;
    auto avgtime = dgemm_time / iterations;
    auto nflops = 2.0 * prk::pow(forder, 3);
    prk::print_flop_rate_time("FP64", nflops/avgtime, avgtime);
  } else {
    std::cout << "Reference checksum = " << reference << "\n"
              << "Actual checksum = " << checksum << std::endl;
#if VERBOSE
    for (int i = 0; i < order; ++i)
      for (int j = 0; j < order; ++j)
        std::cout << "A(" << i << "," << j << ") = " << A[i * order + j] << "\n";
    for (int i = 0; i < order; ++i)
      for (int j = 0; j < order; ++j)
        std::cout << "B(" << i << "," << j << ") = " << B[i * order + j] << "\n";
    for (int i = 0; i < order; ++i)
      for (int j = 0; j < order; ++j)
        std::cout << "C(" << i << "," << j << ") = " << C[i * order + j] << "\n";
    std::cout << std::endl;
#endif
    return 1;
  }

  return 0;
}
