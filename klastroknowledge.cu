// klastroKnowledge.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>    
#include <vector>         
#include <tuple>    

#define CHECK_CUDA(x)        TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)  TORCH_CHECK(x.is_contiguous(),    #x " must be contiguous")
#ifdef CHECK_INPUT
#undef CHECK_INPUT
#endif
#define CHECK_INPUT(x)                                                         \
	CHECK_CUDA(x);                                                             \
	CHECK_CONTIGUOUS(x);                                                       \
	TORCH_CHECK(                                                               \
		(x.scalar_type() == torch::kFloat32) ||                                \
		(x.scalar_type() == torch::kFloat16),                                  \
		#x " must be FP32 or FP16 tensor")


// original feature distance function according to the theory
torch::Tensor compute_inv_covariance(torch::Tensor data, float lambda_reg = 1e-1) {
	CHECK_INPUT(data);  // (N, d)

	auto mean = torch::mean(data, 0, true);          // (1, d)
	auto centered = data - mean;                     // (N, d)
	auto cov = torch::mm(centered.t(), centered) / (data.size(0) - 1);  // (d, d)

	// Regularization: Σ + λI
	auto eye = torch::eye(cov.size(0), cov.options());
	cov += lambda_reg * eye;

	// Inversion
	return torch::linalg_inv(cov);  // or torch::inverse if older libtorch
}


std::tuple<torch::Tensor, torch::Tensor> topk_feature_distance_score(
	torch::Tensor treated,
	torch::Tensor control,
	int k
) {
	CHECK_INPUT(treated);
	CHECK_INPUT(control);

	auto pool = torch::cat({treated, control}, 0);           // (N+M, d)
	auto inv_cov = compute_inv_covariance(pool);             // (d, d)

	auto diff = control.unsqueeze(0) - treated.unsqueeze(1); // (N, M, d)
	auto m2 = torch::einsum("nmd,dd,nmd->nm", {diff, inv_cov, diff});
	auto sim = -m2;  // similarity

	//k = std::min(k, control.size(0));
	k = std::min(k, static_cast<int>(control.size(0)));
	auto [vals, indices] = torch::topk(sim, k, 1, true);  // largest = true

	return std::make_tuple(vals, indices);
}


// new function
std::tuple<torch::Tensor, torch::Tensor> topk_optimized_match(
	torch::Tensor treated,     // (N, d)
	torch::Tensor control,     // (M, d)
	int k,
	bool normalize = true,     // Z-score normalization
	float regularization = 1e-4) {

	
	CHECK_INPUT(treated);
	CHECK_INPUT(control);
	
	const int N = treated.size(0);
	const int M = control.size(0);  
	const int d = treated.size(1);
	
	torch::Tensor treated_proc = treated;
	torch::Tensor control_proc = control;
	
	// ── 1. Z-score normalization
	if (normalize) {
		// normalization with Control values
		auto control_mean = torch::mean(control, 0, true);  // (1, d)
		auto control_std = torch::std(control, 0, true);    // (1, d)
		
		// 0-diversions prevention (if std=0, replace it with a small value)
		control_std = torch::clamp(control_std, 1e-8);
		
		// normalization
		control_proc = (control - control_mean) / control_std;
		treated_proc = (treated - control_mean) / control_std;
	}
	
	// ── 2. get covariance
	auto cov_matrix = torch::cov(control_proc.transpose(0, 1));  // (d, d)
	
	// normalization
	auto eye = torch::eye(d, control.options());
	cov_matrix += eye * regularization;
	
	// inverse matrix
	//auto inv_cov = torch::linalg::inv(cov_matrix);
	auto inv_cov = torch::linalg_inv(cov_matrix);
	
	// A = treated @ inv_cov  (N, d)
	auto A = torch::mm(treated_proc, inv_cov);
	auto B = torch::mm(control_proc, inv_cov);
	
	auto treated_norms = torch::sum(A * treated_proc, 1, true);  // (N, 1)
	auto control_norms = torch::sum(B * control_proc, 1, true);  // (M, 1)
	auto cross_terms = -2.0 * torch::mm(A, control_proc.transpose(0, 1));  // (N, M)
	
	auto distances_sq = treated_norms + control_norms.transpose(0, 1) + cross_terms;  // (N, M)
	auto distances = torch::sqrt(torch::clamp(distances_sq, 1e-8));  // (N, M)
	
	// select top K
	k = std::min(k, M);
	auto [topk_vals, topk_indices] = torch::topk(distances, k, 1, false);
	
	//return {topk_vals, topk_inds};
	return std::make_tuple(topk_vals, topk_indices);   
}

// receive Σ⁻¹ from as as input 
std::tuple<torch::Tensor, torch::Tensor> topk_mahalanobis_with_invcov(
    torch::Tensor treated,
    torch::Tensor control,
    torch::Tensor inv_cov,   // pre-calculated Σ⁻¹
    int k) {

    CHECK_INPUT(treated);
    CHECK_INPUT(control);

    auto diff = control.unsqueeze(0) - treated.unsqueeze(1);  // (N,M,d)
    auto m2   = torch::einsum("nmd,dd,nmd->nm", {diff, inv_cov, diff});
    auto sim  = -m2;

    k = std::min(k, static_cast<int>(control.size(0)));
    auto [vals, indices] = torch::topk(sim, k, 1, true);
    return std::make_tuple(vals, indices);
}

// auto-covariance calculation then batched klastrotest calculation
std::tuple<torch::Tensor, torch::Tensor> batch_topk_optimized_match(
	torch::Tensor treated,
	torch::Tensor control, 
	int k,
	int batch_size = 1000,
	bool normalize = true,
	float regularization = 1e-4) {
	
	CHECK_INPUT(treated);
	CHECK_INPUT(control);
	
	const int N = treated.size(0);

	auto result_vals = torch::zeros({N, k}, torch::dtype(treated.dtype()).device(treated.device()));
	auto result_indices = torch::zeros({N, k}, torch::dtype(torch::kLong).device(treated.device()));
	
	// Process in batches to save memory
	for (int i = 0; i < N; i += batch_size) {
		int end_i = std::min(i + batch_size, N);
		auto treated_batch = treated.slice(0, i, end_i);
		//auto treated_batch = treated.slice(/*dim=*/0, /*start=*/i, /*end=*/end_i);

		auto batch_out = topk_optimized_match(
							treated_batch, control, k, normalize, regularization);

		auto batch_vals    = std::get<0>(batch_out);
		auto batch_indices = std::get<1>(batch_out);

		result_vals.slice   (0, i, end_i) = batch_vals;
		result_indices.slice(0, i, end_i) = batch_indices;
		
		//auto batch_indices = topk_optimized_match(treated_batch, control, k, normalize, regularization);
		//result_indices.slice(0, i, end_i) = batch_indices;
	}
	
	//return result_indices;
	return std::make_tuple(result_vals, result_indices); 
}


torch::Tensor softmax_entropy(torch::Tensor vals) {
	CHECK_INPUT(vals);
	auto log_p = torch::log_softmax(vals, /*dim=*/1);
	auto p     = log_p.exp();
	return -(p * log_p).sum(1).mean();          // scalar
}

torch::Tensor kmargin(torch::Tensor vals) {
	CHECK_INPUT(vals);
	auto top2 = std::get<0>(vals.topk(2, 1));
	return (top2.select(1,0) - top2.select(1,1)).mean();  // scalar
}

torch::Tensor kmargin_n(torch::Tensor vals, int64_t rank_n) {
	CHECK_INPUT(vals);
	TORCH_CHECK(rank_n >= 2, "rank_n must be >= 2");

	// topk returns (values, indices)
	auto topk = std::get<0>(vals.topk(rank_n, /*dim=*/1));   // (B, rank_n)

	// top-1 ([:,0])  minus top-N ([, rank_n-1])
	auto margin_n = (topk.select(1, 0) - topk.select(1, rank_n - 1)).mean();
	return margin_n;   // scalar tensor
}


torch::Tensor kmargin_avg(torch::Tensor vals, int64_t n_tail=10) {
	CHECK_INPUT(vals);
	auto topk = std::get<0>(vals.topk(n_tail, 1));        
	auto top1 = topk.select(1, 0);                        
	auto tail_mean = topk.slice(1, 1).mean(1);             
	return (top1 - tail_mean).mean();                     
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

	m.def("topk_mahalanobis_with_invcov",
		&topk_mahalanobis_with_invcov,
		"Mahalanobis top-k with precomputed Σ⁻¹",
		py::arg("treated"), py::arg("control"),
		py::arg("inv_cov"), py::arg("k"));

	m.def("compute_inv_covariance",
		&compute_inv_covariance,
		"Compute inverse covariance matrix with regularization",
		py::arg("data"),
		py::arg("lambda_reg") = 1e-1);

	m.def("topk_feature_distance_score",
		&topk_feature_distance_score,
		"Auto-covariance Mahalanobis top-k matcher (returns vals, idx)",
		py::arg("treated"),
		py::arg("control"),
		py::arg("k"));


	m.def("topk_optimized_match",
		&topk_optimized_match,
		"klastroKnowledge distance top-k w/ covariance (returns vals, idx)",
		py::arg("treated"), py::arg("control"), py::arg("k"),
		py::arg("normalize") = true, py::arg("regularization") = 1e-4);

	m.def("batch_topk_optimized_match",
		&batch_topk_optimized_match,
		"Batched klastroKnowledge distance top-k (returns vals, idx)",
		py::arg("treated"), py::arg("control"), py::arg("k"),
		py::arg("batch_size") = 1000, py::arg("normalize") = true,
		py::arg("regularization") = 1e-4);


	m.def("softmax_entropy", &softmax_entropy,
		  "Softmax-based information entropy (scalar)");
	m.def("kmargin",          &kmargin,
		  "Mean top-1 minus top-2 margin (scalar)");
	m.def(
		"kmargin_n",
		&kmargin_n,                                
		"Top-1 minus Top-n mean margin (n ≥ 2)",   
		py::arg("vals"),                             
		py::arg("rank_n")                          
	);

	m.def(
		"kmargin_avg",
		&kmargin_avg,
		"Top-1 minus mean(Top-2 … Top-n) margin",
		py::arg("vals"),
		py::arg("n_tail") = 10        
	);
}
			

