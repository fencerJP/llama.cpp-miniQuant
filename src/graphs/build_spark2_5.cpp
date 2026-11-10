#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_spark2_5() {
    ggml_cgraph * gf = new_graph_custom();

    ggml_tensor * inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    ggml_tensor * KQ_mask_swa = hparams.n_swa > 0
        ? kv_self.any_compacted()
            ? build_swa_mask_for_graph(hparams.n_swa, true)
            : build_inp_KQ_mask_swa()
        : nullptr;

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));

    for (int il = 0; il < n_layer; ++il) {
        const bool is_swa = hparams.swa_layers[il];
        const int n_swa_l = is_swa ? hparams.n_swa : 0;

        auto KQ_mask_l = is_swa ? KQ_mask_swa : KQ_mask;
        GGML_ASSERT(KQ_mask_l != nullptr);
        auto rope_factors = is_swa ? nullptr : build_rope_factors(il);

        const bool is_final_layer = il == n_layer - 1;
        ggml_tensor * attn_out_ids = is_final_layer ? inp_out_ids : nullptr;

        auto cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                        inp_pos, attn_out_ids, rope_factors,
                        KQ_mask_l, nullptr, nullptr, kq_scale, 0.0f, n_swa_l, il, true, false, true);

        // Spark2.5 uses GELU activation for dense MLP
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, cb, il, gf, true);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
