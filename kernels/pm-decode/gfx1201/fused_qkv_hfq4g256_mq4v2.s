.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected fused_qkv_mq4g256v2
.globl fused_qkv_mq4g256v2
.p2align 8
.type fused_qkv_mq4g256v2,@function
fused_qkv_mq4g256v2:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b128 s[8:11], s[0:1], 0x10
	s_load_b128 s[12:15], s[0:1], 0x20
	s_load_b128 s[16:19], s[0:1], 0x30
	s_load_b64 s[20:21], s[0:1], 0x40
	s_wait_kmcnt 0x0
	s_add_co_i32 s22, s18, s19
	s_add_co_i32 s23, s22, s20
	s_cmp_ge_i32 ttmp9, s23
	s_cbranch_scc1 .Lqkv_done
	s_sub_co_i32 s30, ttmp9, s18
	s_cmp_lt_i32 ttmp9, s18
	s_cselect_b32 s24, ttmp9, s30
	s_cselect_b32 s26, s4, s6
	s_cselect_b32 s27, s5, s7
	s_cselect_b32 s28, s12, s14
	s_cselect_b32 s29, s13, s15
	s_sub_co_i32 s30, ttmp9, s22
	s_cmp_lt_i32 ttmp9, s22
	s_cselect_b32 s24, s24, s30
	s_cselect_b32 s26, s26, s8
	s_cselect_b32 s27, s27, s9
	s_cselect_b32 s28, s28, s16
	s_cselect_b32 s29, s29, s17
	s_ashr_i32 s25, s24, 31
	s_ashr_i32 s31, s21, 8
	s_lshr_b32 s34, s31, 2
	s_and_b32 s35, s31, 3
	s_mul_i32 s32, s31, 0x88
	s_ashr_i32 s33, s32, 31
	s_mul_u64 s[32:33], s[32:33], s[24:25]
	s_add_nc_u64 s[32:33], s[26:27], s[32:33]
	s_mov_b64 s[36:37], s[32:33]
	s_and_b32 s37, s37, 0xffff
	s_mov_b32 s38, -1
	s_mov_b32 s39, 0x31004000
	s_mov_b32 s40, 0
	v_lshlrev_b32_e32 v1, 2, v0
	v_add_nc_u32_e32 v1, 8, v1
	v_lshlrev_b32_e32 v2, 5, v0
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	v_cndmask_b32_e64 v3, 4, 0, vcc_lo
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	s_cmp_eq_u32 s34, 0
	s_cbranch_scc1 .Lqkv_tails
	.Lqkv_quads:
	buffer_load_b64 v[8:9], v31, s[36:39], s40 offen scope:SCOPE_DEV
	buffer_load_b32 v4, v1, s[36:39], s40 offen scope:SCOPE_DEV
	buffer_load_b32 v10, v31, s[36:39], s40 offen offset:136 scope:SCOPE_DEV
	buffer_load_b32 v11, v31, s[36:39], s40 offen offset:140 scope:SCOPE_DEV
	buffer_load_b32 v5, v1, s[36:39], s40 offen offset:136 scope:SCOPE_DEV
	buffer_load_b32 v12, v31, s[36:39], s40 offen offset:272 scope:SCOPE_DEV
	buffer_load_b32 v13, v31, s[36:39], s40 offen offset:276 scope:SCOPE_DEV
	buffer_load_b32 v6, v1, s[36:39], s40 offen offset:272 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v31, s[36:39], s40 offen offset:408 scope:SCOPE_DEV
	buffer_load_b32 v7, v1, s[36:39], s40 offen offset:408 scope:SCOPE_DEV
	global_load_b128 v[32:35], v2, s[10:11]
	global_load_b128 v[40:43], v2, s[10:11] offset:1024
	global_load_b128 v[48:51], v2, s[10:11] offset:2048
	global_load_b128 v[56:59], v2, s[10:11] offset:3072
	global_load_b128 v[36:39], v2, s[10:11] offset:16
	global_load_b128 v[44:47], v2, s[10:11] offset:1040
	global_load_b128 v[52:55], v2, s[10:11] offset:2064
	global_load_b128 v[60:63], v2, s[10:11] offset:3088
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x11
	v_cndmask_b32_e64 v8, v9, v8, vcc_lo
	s_wait_loadcnt 0xe
	v_cndmask_b32_e64 v10, v11, v10, vcc_lo
	s_wait_loadcnt 0xb
	v_cndmask_b32_e64 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x9
	v_cndmask_b32_e64 v14, v15, v14, vcc_lo
	v_bfe_u32 v20, v4, 4, 4
	v_bfe_u32 v22, v6, 4, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v22, v22
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v12, v22, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_mul_f32_e32 v16, v20, v33
	s_wait_loadcnt 0x5
	v_mul_f32_e32 v18, v22, v49
	v_bfe_u32 v20, v4, 0, 4
	v_bfe_u32 v21, v5, 4, 4
	v_bfe_u32 v22, v6, 0, 4
	v_bfe_u32 v23, v7, 4, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v10, v21, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v12, v22, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v14, v23, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v16, v20, v32 :: v_dual_mul_f32 v17, v21, v41
	s_delay_alu instid0(VALU_DEP_2)
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v18, v22, v48 :: v_dual_mul_f32 v19, v23, v57
	v_bfe_u32 v20, v4, 8, 4
	v_bfe_u32 v21, v5, 0, 4
	v_bfe_u32 v22, v6, 8, 4
	v_bfe_u32 v23, v7, 0, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v10, v21, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v12, v22, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v14, v23, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v16, v20, v34 :: v_dual_fmac_f32 v17, v21, v40
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v18, v22, v50 :: v_dual_fmac_f32 v19, v23, v56
	v_bfe_u32 v20, v4, 12, 4
	v_bfe_u32 v21, v5, 8, 4
	v_bfe_u32 v22, v6, 12, 4
	v_bfe_u32 v23, v7, 8, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v10, v21, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v12, v22, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v14, v23, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v16, v20, v35 :: v_dual_fmac_f32 v17, v21, v42
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v18, v22, v51 :: v_dual_fmac_f32 v19, v23, v58
	v_bfe_u32 v20, v4, 16, 4
	v_bfe_u32 v21, v5, 12, 4
	v_bfe_u32 v22, v6, 16, 4
	v_bfe_u32 v23, v7, 12, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v10, v21, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v12, v22, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v14, v23, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_3)
	s_wait_loadcnt 0x3
	v_dual_fmac_f32 v16, v20, v36 :: v_dual_fmac_f32 v17, v21, v43
	s_delay_alu instid0(VALU_DEP_2)
	s_wait_loadcnt 0x1
	v_dual_fmac_f32 v18, v22, v52 :: v_dual_fmac_f32 v19, v23, v59
	v_bfe_u32 v20, v4, 20, 4
	v_bfe_u32 v21, v5, 16, 4
	v_bfe_u32 v22, v6, 20, 4
	v_bfe_u32 v23, v7, 16, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v10, v21, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v12, v22, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v14, v23, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v16, v20, v37 :: v_dual_fmac_f32 v17, v21, v44
	s_delay_alu instid0(VALU_DEP_2)
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v18, v22, v53 :: v_dual_fmac_f32 v19, v23, v60
	v_bfe_u32 v20, v4, 24, 4
	v_bfe_u32 v21, v5, 20, 4
	v_bfe_u32 v22, v6, 24, 4
	v_bfe_u32 v23, v7, 20, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v10, v21, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v12, v22, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v14, v23, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v16, v20, v38 :: v_dual_fmac_f32 v17, v21, v45
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v18, v22, v54 :: v_dual_fmac_f32 v19, v23, v61
	v_bfe_u32 v20, v4, 28, 4
	v_bfe_u32 v21, v5, 24, 4
	v_bfe_u32 v22, v6, 28, 4
	v_bfe_u32 v23, v7, 24, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v10, v21, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v12, v22, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v14, v23, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v16, v20, v39 :: v_dual_fmac_f32 v17, v21, v46
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v18, v22, v55 :: v_dual_fmac_f32 v19, v23, v62
	v_bfe_u32 v21, v5, 28, 4
	v_bfe_u32 v23, v7, 28, 4
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v21, v10, v21, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v14, v23, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v17, v21, v47
	v_fmac_f32_e32 v19, v23, v63
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v26, v26, v16 :: v_dual_add_f32 v27, v27, v17
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v28, v28, v18 :: v_dual_add_f32 v29, v29, v19
	s_add_co_i32 s40, s40, 0x220
	v_add_nc_u32_e32 v2, 0x1000, v2
	s_sub_co_i32 s34, s34, 1
	s_cmp_lg_u32 s34, 0
	s_cbranch_scc1 .Lqkv_quads
	.Lqkv_tails:
	s_cmp_lt_u32 s35, 1
	s_cbranch_scc1 .Lqkv_reduce
	v_add_nc_u32_e32 v4, s40, v3
	v_add_nc_u32_e32 v7, s40, v1
	global_load_b32 v5, v4, s[32:33]
	global_load_b32 v6, v7, s[32:33]
	global_load_b128 v[8:11], v2, s[10:11]
	global_load_b128 v[12:15], v2, s[10:11] offset:16
	s_wait_loadcnt 0x2
	v_bfe_u32 v16, v6, 0, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_fma_mix_f32 v16, v5, v16, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v6, 4, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_fma_mix_f32 v17, v5, v17, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v6, 8, 4
	v_cvt_f32_ubyte0_e32 v18, v18
	v_fma_mix_f32 v18, v5, v18, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v6, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v19, v5, v19, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v6, 16, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_fma_mix_f32 v20, v5, v20, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v6, 20, 4
	v_cvt_f32_ubyte0_e32 v21, v21
	v_fma_mix_f32 v21, v5, v21, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v6, 24, 4
	v_cvt_f32_ubyte0_e32 v22, v22
	v_fma_mix_f32 v22, v5, v22, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v6, 28, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v23, v5, v23, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v24, v17, v9
	v_fmac_f32_e32 v24, v16, v8
	v_fmac_f32_e32 v24, v18, v10
	v_fmac_f32_e32 v24, v19, v11
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v24, v20, v12
	v_fmac_f32_e32 v24, v21, v13
	v_fmac_f32_e32 v24, v22, v14
	v_fmac_f32_e32 v24, v23, v15
	v_add_f32_e32 v26, v26, v24
	s_cmp_lt_u32 s35, 2
	s_cbranch_scc1 .Lqkv_reduce
	v_add_nc_u32_e32 v4, s40, v3
	v_add_nc_u32_e32 v7, s40, v1
	global_load_b32 v5, v4, s[32:33] offset:136
	global_load_b32 v6, v7, s[32:33] offset:136
	global_load_b128 v[8:11], v2, s[10:11] offset:1024
	global_load_b128 v[12:15], v2, s[10:11] offset:1040
	s_wait_loadcnt 0x2
	v_bfe_u32 v16, v6, 0, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_fma_mix_f32 v16, v5, v16, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v6, 4, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_fma_mix_f32 v17, v5, v17, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v6, 8, 4
	v_cvt_f32_ubyte0_e32 v18, v18
	v_fma_mix_f32 v18, v5, v18, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v6, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v19, v5, v19, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v6, 16, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_fma_mix_f32 v20, v5, v20, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v6, 20, 4
	v_cvt_f32_ubyte0_e32 v21, v21
	v_fma_mix_f32 v21, v5, v21, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v6, 24, 4
	v_cvt_f32_ubyte0_e32 v22, v22
	v_fma_mix_f32 v22, v5, v22, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v6, 28, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v23, v5, v23, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v24, v17, v9
	v_fmac_f32_e32 v24, v16, v8
	v_fmac_f32_e32 v24, v18, v10
	v_fmac_f32_e32 v24, v19, v11
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v24, v20, v12
	v_fmac_f32_e32 v24, v21, v13
	v_fmac_f32_e32 v24, v22, v14
	v_fmac_f32_e32 v24, v23, v15
	v_add_f32_e32 v27, v27, v24
	s_cmp_lt_u32 s35, 3
	s_cbranch_scc1 .Lqkv_reduce
	v_add_nc_u32_e32 v4, s40, v3
	v_add_nc_u32_e32 v7, s40, v1
	global_load_b32 v5, v4, s[32:33] offset:272
	global_load_b32 v6, v7, s[32:33] offset:272
	global_load_b128 v[8:11], v2, s[10:11] offset:2048
	global_load_b128 v[12:15], v2, s[10:11] offset:2064
	s_wait_loadcnt 0x2
	v_bfe_u32 v16, v6, 0, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_fma_mix_f32 v16, v5, v16, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v6, 4, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_fma_mix_f32 v17, v5, v17, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v6, 8, 4
	v_cvt_f32_ubyte0_e32 v18, v18
	v_fma_mix_f32 v18, v5, v18, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v6, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v19, v5, v19, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v6, 16, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_fma_mix_f32 v20, v5, v20, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v6, 20, 4
	v_cvt_f32_ubyte0_e32 v21, v21
	v_fma_mix_f32 v21, v5, v21, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v6, 24, 4
	v_cvt_f32_ubyte0_e32 v22, v22
	v_fma_mix_f32 v22, v5, v22, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v6, 28, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v23, v5, v23, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v24, v17, v9
	v_fmac_f32_e32 v24, v16, v8
	v_fmac_f32_e32 v24, v18, v10
	v_fmac_f32_e32 v24, v19, v11
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v24, v20, v12
	v_fmac_f32_e32 v24, v21, v13
	v_fmac_f32_e32 v24, v22, v14
	v_fmac_f32_e32 v24, v23, v15
	v_add_f32_e32 v28, v28, v24
	.Lqkv_reduce:
	v_add_f32_e32 v24, v26, v27
	v_add_f32_e32 v25, v28, v29
	v_add_f32_e32 v24, v24, v25
	ds_swizzle_b32 v30, v24 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v30
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v31, 0, 8, vcc_lo
	v_add_lshl_u32 v31, v31, v0, 2
	ds_bpermute_b32 v30, v31, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v30
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v31, 0, 4, vcc_lo
	v_add_lshl_u32 v31, v31, v0, 2
	ds_bpermute_b32 v30, v31, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v30
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v31, 0, 2, vcc_lo
	v_add_lshl_u32 v31, v31, v0, 2
	ds_bpermute_b32 v30, v31, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v30
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v31, 0, 1, vcc_lo
	v_add_lshl_u32 v31, v31, v0, 2
	ds_bpermute_b32 v30, v31, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v30
	v_cmpx_eq_u32_e32 0, v0
	s_cbranch_execz .Lqkv_done
	s_lshl_b64 s[24:25], s[24:25], 2
	s_add_nc_u64 s[32:33], s[28:29], s[24:25]
	v_mov_b32_e32 v1, 0
	global_store_b32 v1, v24, s[32:33]
	s_wait_storecnt 0x0
	.Lqkv_done:
	s_endpgm
.Lfused_qkv_mq4g256v2_end:
.size fused_qkv_mq4g256v2, .Lfused_qkv_mq4g256v2_end-fused_qkv_mq4g256v2
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel fused_qkv_mq4g256v2
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 72
	.amdhsa_user_sgpr_count 2
	.amdhsa_user_sgpr_dispatch_ptr 0
	.amdhsa_user_sgpr_queue_ptr 0
	.amdhsa_user_sgpr_kernarg_segment_ptr 1
	.amdhsa_user_sgpr_dispatch_id 0
	.amdhsa_user_sgpr_private_segment_size 0
	.amdhsa_wavefront_size32 1
	.amdhsa_uses_dynamic_stack 0
	.amdhsa_enable_private_segment 0
	.amdhsa_system_sgpr_workgroup_id_x 1
	.amdhsa_system_sgpr_workgroup_id_y 0
	.amdhsa_system_sgpr_workgroup_id_z 0
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr 64
	.amdhsa_next_free_sgpr 42
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lfused_qkv_mq4g256v2_end-fused_qkv_mq4g256v2)<<4)&4080)>>4
	.amdhsa_round_robin_scheduling 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A_q
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A_k
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A_v
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y_q
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y_k
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y_v
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - 
        .name: q_m
        .offset: 56
        .size: 4
        .value_kind: by_value
      - 
        .name: k_m
        .offset: 60
        .size: 4
        .value_kind: by_value
      - 
        .name: v_m
        .offset: 64
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 68
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .max_flat_workgroup_size: 32
    .name: fused_qkv_mq4g256v2
    .private_segment_fixed_size: 0
    .sgpr_count: 44
    .sgpr_spill_count: 0
    .symbol: fused_qkv_mq4g256v2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 64
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
