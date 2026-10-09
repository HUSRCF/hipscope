.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected gemv_mq4g256v2_residual_xbatch
.globl gemv_mq4g256v2_residual_xbatch
.p2align 8
.type gemv_mq4g256v2_residual_xbatch,@function
gemv_mq4g256v2_residual_xbatch:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b128 s[8:11], s[0:1], 0x10
	s_load_b32 s3, s[0:1], 0x20
	s_wait_kmcnt 0x0
	s_lshr_b32 s14, s11, 8
	s_mul_i32 s19, s14, 0x88
	s_mov_b32 s20, s4
	s_and_b32 s21, s5, 0xffff
	s_mov_b32 s22, -1
	s_mov_b32 s23, 0x31004000
	s_lshl_b32 s44, s11, 2
	s_lshl_b32 s45, s10, 2
	s_mov_b32 s28, s6
	s_mov_b32 s29, s7
	s_mul_i32 s24, s44, 1
	s_add_co_u32 s30, s6, s24
	s_add_co_ci_u32 s31, s7, 0
	s_mul_i32 s24, s44, 2
	s_add_co_u32 s32, s6, s24
	s_add_co_ci_u32 s33, s7, 0
	s_mul_i32 s24, s44, 3
	s_add_co_u32 s34, s6, s24
	s_add_co_ci_u32 s35, s7, 0
	s_mul_i32 s24, s44, 4
	s_add_co_u32 s36, s6, s24
	s_add_co_ci_u32 s37, s7, 0
	s_mul_i32 s24, s44, 5
	s_add_co_u32 s38, s6, s24
	s_add_co_ci_u32 s39, s7, 0
	s_mul_i32 s24, s44, 6
	s_add_co_u32 s40, s6, s24
	s_add_co_ci_u32 s41, s7, 0
	s_mul_i32 s24, s44, 7
	s_add_co_u32 s42, s6, s24
	s_add_co_ci_u32 s43, s7, 0
	v_lshlrev_b32_e32 v1, 2, v0
	s_cmp_eq_u32 s3, 1
	s_cbranch_scc1 .Lrx_b1
	s_cmp_eq_u32 s3, 2
	s_cbranch_scc1 .Lrx_b2
	s_cmp_eq_u32 s3, 3
	s_cbranch_scc1 .Lrx_b3
	s_cmp_eq_u32 s3, 4
	s_cbranch_scc1 .Lrx_b4
	s_cmp_eq_u32 s3, 5
	s_cbranch_scc1 .Lrx_b5
	s_cmp_eq_u32 s3, 6
	s_cbranch_scc1 .Lrx_b6
	s_cmp_eq_u32 s3, 7
	s_cbranch_scc1 .Lrx_b7
	s_cmp_eq_u32 s3, 8
	s_cbranch_scc1 .Lrx_b8
	s_branch .Lrx_end
	.Lrx_b1:
	s_mul_i32 s12, ttmp9, 8
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_mul_i32 s13, s13, s19
	v_mov_b32_e32 v12, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v13, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v14, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v15, s13
	v_add_nc_u32_e32 v7, s13, v1
	s_add_co_i32 s13, s12, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v16, s13
	v_add_nc_u32_e32 v8, s13, v1
	s_add_co_i32 s13, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v17, s13
	v_add_nc_u32_e32 v9, s13, v1
	s_add_co_i32 s13, s12, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v18, s13
	v_add_nc_u32_e32 v10, s13, v1
	s_add_co_i32 s13, s12, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v19, s13
	v_add_nc_u32_e32 v11, s13, v1
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s0_done
	s_mov_b32 s15, 0x0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[148:151], v2, s[28:29]
	global_load_b128 v[152:155], v2, s[28:29] offset:16
	.Lrx_b1_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0x6
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0x4
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x2
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v44, v148 :: v_dual_mul_f32 v53, v55, v149
	v_dual_mul_f32 v62, v64, v148 :: v_dual_mul_f32 v63, v75, v149
	v_dual_mul_f32 v72, v84, v148 :: v_dual_mul_f32 v73, v95, v149
	v_dual_mul_f32 v82, v104, v148 :: v_dual_mul_f32 v83, v115, v149
	v_dual_fmac_f32 v52, v45, v149 :: v_dual_fmac_f32 v53, v54, v148
	v_dual_fmac_f32 v62, v65, v149 :: v_dual_fmac_f32 v63, v74, v148
	v_dual_fmac_f32 v72, v85, v149 :: v_dual_fmac_f32 v73, v94, v148
	v_dual_fmac_f32 v82, v105, v149 :: v_dual_fmac_f32 v83, v114, v148
	v_dual_fmac_f32 v52, v46, v150 :: v_dual_fmac_f32 v53, v56, v150
	v_dual_fmac_f32 v62, v66, v150 :: v_dual_fmac_f32 v63, v76, v150
	v_dual_fmac_f32 v72, v86, v150 :: v_dual_fmac_f32 v73, v96, v150
	v_dual_fmac_f32 v82, v106, v150 :: v_dual_fmac_f32 v83, v116, v150
	v_dual_fmac_f32 v52, v47, v151 :: v_dual_fmac_f32 v53, v57, v151
	v_dual_fmac_f32 v62, v67, v151 :: v_dual_fmac_f32 v63, v77, v151
	v_dual_fmac_f32 v72, v87, v151 :: v_dual_fmac_f32 v73, v97, v151
	v_dual_fmac_f32 v82, v107, v151 :: v_dual_fmac_f32 v83, v117, v151
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v52, v48, v152 :: v_dual_fmac_f32 v53, v58, v152
	v_dual_fmac_f32 v62, v68, v152 :: v_dual_fmac_f32 v63, v78, v152
	v_dual_fmac_f32 v72, v88, v152 :: v_dual_fmac_f32 v73, v98, v152
	v_dual_fmac_f32 v82, v108, v152 :: v_dual_fmac_f32 v83, v118, v152
	v_dual_fmac_f32 v52, v49, v153 :: v_dual_fmac_f32 v53, v59, v153
	v_dual_fmac_f32 v62, v69, v153 :: v_dual_fmac_f32 v63, v79, v153
	v_dual_fmac_f32 v72, v89, v153 :: v_dual_fmac_f32 v73, v99, v153
	v_dual_fmac_f32 v82, v109, v153 :: v_dual_fmac_f32 v83, v119, v153
	v_dual_fmac_f32 v52, v50, v154 :: v_dual_fmac_f32 v53, v60, v154
	v_dual_fmac_f32 v62, v70, v154 :: v_dual_fmac_f32 v63, v80, v154
	v_dual_fmac_f32 v72, v90, v154 :: v_dual_fmac_f32 v73, v100, v154
	v_dual_fmac_f32 v82, v110, v154 :: v_dual_fmac_f32 v83, v120, v154
	v_dual_fmac_f32 v52, v51, v155 :: v_dual_fmac_f32 v53, v61, v155
	v_dual_fmac_f32 v62, v71, v155 :: v_dual_fmac_f32 v63, v81, v155
	v_dual_fmac_f32 v72, v91, v155 :: v_dual_fmac_f32 v73, v101, v155
	v_dual_fmac_f32 v82, v111, v155 :: v_dual_fmac_f32 v83, v121, v155
	global_load_b128 v[148:151], v3, s[28:29]
	global_load_b128 v[152:155], v3, s[28:29] offset:16
	v_dual_add_f32 v124, v52, v124 :: v_dual_add_f32 v125, v125, v53
	v_dual_add_f32 v126, v62, v126 :: v_dual_add_f32 v127, v127, v63
	v_dual_add_f32 v128, v72, v128 :: v_dual_add_f32 v129, v129, v73
	v_dual_add_f32 v130, v82, v130 :: v_dual_add_f32 v131, v131, v83
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s0
	s_wait_loadcnt 0x0
	.Lrx_b1_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[148:151], v2, s[28:29]
	global_load_b128 v[152:155], v2, s[28:29] offset:16
	.Lrx_b1_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0x6
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0x4
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x2
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v44, v148 :: v_dual_mul_f32 v53, v55, v149
	v_dual_mul_f32 v62, v64, v148 :: v_dual_mul_f32 v63, v75, v149
	v_dual_mul_f32 v72, v84, v148 :: v_dual_mul_f32 v73, v95, v149
	v_dual_mul_f32 v82, v104, v148 :: v_dual_mul_f32 v83, v115, v149
	v_dual_fmac_f32 v52, v45, v149 :: v_dual_fmac_f32 v53, v54, v148
	v_dual_fmac_f32 v62, v65, v149 :: v_dual_fmac_f32 v63, v74, v148
	v_dual_fmac_f32 v72, v85, v149 :: v_dual_fmac_f32 v73, v94, v148
	v_dual_fmac_f32 v82, v105, v149 :: v_dual_fmac_f32 v83, v114, v148
	v_dual_fmac_f32 v52, v46, v150 :: v_dual_fmac_f32 v53, v56, v150
	v_dual_fmac_f32 v62, v66, v150 :: v_dual_fmac_f32 v63, v76, v150
	v_dual_fmac_f32 v72, v86, v150 :: v_dual_fmac_f32 v73, v96, v150
	v_dual_fmac_f32 v82, v106, v150 :: v_dual_fmac_f32 v83, v116, v150
	v_dual_fmac_f32 v52, v47, v151 :: v_dual_fmac_f32 v53, v57, v151
	v_dual_fmac_f32 v62, v67, v151 :: v_dual_fmac_f32 v63, v77, v151
	v_dual_fmac_f32 v72, v87, v151 :: v_dual_fmac_f32 v73, v97, v151
	v_dual_fmac_f32 v82, v107, v151 :: v_dual_fmac_f32 v83, v117, v151
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v52, v48, v152 :: v_dual_fmac_f32 v53, v58, v152
	v_dual_fmac_f32 v62, v68, v152 :: v_dual_fmac_f32 v63, v78, v152
	v_dual_fmac_f32 v72, v88, v152 :: v_dual_fmac_f32 v73, v98, v152
	v_dual_fmac_f32 v82, v108, v152 :: v_dual_fmac_f32 v83, v118, v152
	v_dual_fmac_f32 v52, v49, v153 :: v_dual_fmac_f32 v53, v59, v153
	v_dual_fmac_f32 v62, v69, v153 :: v_dual_fmac_f32 v63, v79, v153
	v_dual_fmac_f32 v72, v89, v153 :: v_dual_fmac_f32 v73, v99, v153
	v_dual_fmac_f32 v82, v109, v153 :: v_dual_fmac_f32 v83, v119, v153
	v_dual_fmac_f32 v52, v50, v154 :: v_dual_fmac_f32 v53, v60, v154
	v_dual_fmac_f32 v62, v70, v154 :: v_dual_fmac_f32 v63, v80, v154
	v_dual_fmac_f32 v72, v90, v154 :: v_dual_fmac_f32 v73, v100, v154
	v_dual_fmac_f32 v82, v110, v154 :: v_dual_fmac_f32 v83, v120, v154
	v_dual_fmac_f32 v52, v51, v155 :: v_dual_fmac_f32 v53, v61, v155
	v_dual_fmac_f32 v62, v71, v155 :: v_dual_fmac_f32 v63, v81, v155
	v_dual_fmac_f32 v72, v91, v155 :: v_dual_fmac_f32 v73, v101, v155
	v_dual_fmac_f32 v82, v111, v155 :: v_dual_fmac_f32 v83, v121, v155
	global_load_b128 v[148:151], v3, s[28:29]
	global_load_b128 v[152:155], v3, s[28:29] offset:16
	v_dual_add_f32 v132, v52, v132 :: v_dual_add_f32 v133, v133, v53
	v_dual_add_f32 v134, v62, v134 :: v_dual_add_f32 v135, v135, v63
	v_dual_add_f32 v136, v72, v136 :: v_dual_add_f32 v137, v137, v73
	v_dual_add_f32 v138, v82, v138 :: v_dual_add_f32 v139, v139, v83
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s1
	s_wait_loadcnt 0x0
	.Lrx_b1_s1_done:
	v_dual_add_f32 v124, v124, v132 :: v_dual_add_f32 v125, v125, v133
	v_dual_add_f32 v126, v126, v134 :: v_dual_add_f32 v127, v127, v135
	v_dual_add_f32 v128, v128, v136 :: v_dual_add_f32 v129, v129, v137
	v_dual_add_f32 v130, v130, v138 :: v_dual_add_f32 v131, v131, v139
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[148:151], v2, s[28:29]
	global_load_b128 v[152:155], v2, s[28:29] offset:16
	.Lrx_b1_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0x6
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0x4
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x2
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v44, v148 :: v_dual_mul_f32 v53, v55, v149
	v_dual_mul_f32 v62, v64, v148 :: v_dual_mul_f32 v63, v75, v149
	v_dual_mul_f32 v72, v84, v148 :: v_dual_mul_f32 v73, v95, v149
	v_dual_mul_f32 v82, v104, v148 :: v_dual_mul_f32 v83, v115, v149
	v_dual_fmac_f32 v52, v45, v149 :: v_dual_fmac_f32 v53, v54, v148
	v_dual_fmac_f32 v62, v65, v149 :: v_dual_fmac_f32 v63, v74, v148
	v_dual_fmac_f32 v72, v85, v149 :: v_dual_fmac_f32 v73, v94, v148
	v_dual_fmac_f32 v82, v105, v149 :: v_dual_fmac_f32 v83, v114, v148
	v_dual_fmac_f32 v52, v46, v150 :: v_dual_fmac_f32 v53, v56, v150
	v_dual_fmac_f32 v62, v66, v150 :: v_dual_fmac_f32 v63, v76, v150
	v_dual_fmac_f32 v72, v86, v150 :: v_dual_fmac_f32 v73, v96, v150
	v_dual_fmac_f32 v82, v106, v150 :: v_dual_fmac_f32 v83, v116, v150
	v_dual_fmac_f32 v52, v47, v151 :: v_dual_fmac_f32 v53, v57, v151
	v_dual_fmac_f32 v62, v67, v151 :: v_dual_fmac_f32 v63, v77, v151
	v_dual_fmac_f32 v72, v87, v151 :: v_dual_fmac_f32 v73, v97, v151
	v_dual_fmac_f32 v82, v107, v151 :: v_dual_fmac_f32 v83, v117, v151
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v52, v48, v152 :: v_dual_fmac_f32 v53, v58, v152
	v_dual_fmac_f32 v62, v68, v152 :: v_dual_fmac_f32 v63, v78, v152
	v_dual_fmac_f32 v72, v88, v152 :: v_dual_fmac_f32 v73, v98, v152
	v_dual_fmac_f32 v82, v108, v152 :: v_dual_fmac_f32 v83, v118, v152
	v_dual_fmac_f32 v52, v49, v153 :: v_dual_fmac_f32 v53, v59, v153
	v_dual_fmac_f32 v62, v69, v153 :: v_dual_fmac_f32 v63, v79, v153
	v_dual_fmac_f32 v72, v89, v153 :: v_dual_fmac_f32 v73, v99, v153
	v_dual_fmac_f32 v82, v109, v153 :: v_dual_fmac_f32 v83, v119, v153
	v_dual_fmac_f32 v52, v50, v154 :: v_dual_fmac_f32 v53, v60, v154
	v_dual_fmac_f32 v62, v70, v154 :: v_dual_fmac_f32 v63, v80, v154
	v_dual_fmac_f32 v72, v90, v154 :: v_dual_fmac_f32 v73, v100, v154
	v_dual_fmac_f32 v82, v110, v154 :: v_dual_fmac_f32 v83, v120, v154
	v_dual_fmac_f32 v52, v51, v155 :: v_dual_fmac_f32 v53, v61, v155
	v_dual_fmac_f32 v62, v71, v155 :: v_dual_fmac_f32 v63, v81, v155
	v_dual_fmac_f32 v72, v91, v155 :: v_dual_fmac_f32 v73, v101, v155
	v_dual_fmac_f32 v82, v111, v155 :: v_dual_fmac_f32 v83, v121, v155
	global_load_b128 v[148:151], v3, s[28:29]
	global_load_b128 v[152:155], v3, s[28:29] offset:16
	v_dual_add_f32 v132, v52, v132 :: v_dual_add_f32 v133, v133, v53
	v_dual_add_f32 v134, v62, v134 :: v_dual_add_f32 v135, v135, v63
	v_dual_add_f32 v136, v72, v136 :: v_dual_add_f32 v137, v137, v73
	v_dual_add_f32 v138, v82, v138 :: v_dual_add_f32 v139, v139, v83
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s2
	s_wait_loadcnt 0x0
	.Lrx_b1_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[148:151], v2, s[28:29]
	global_load_b128 v[152:155], v2, s[28:29] offset:16
	.Lrx_b1_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0x6
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0x4
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x2
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v44, v148 :: v_dual_mul_f32 v53, v55, v149
	v_dual_mul_f32 v62, v64, v148 :: v_dual_mul_f32 v63, v75, v149
	v_dual_mul_f32 v72, v84, v148 :: v_dual_mul_f32 v73, v95, v149
	v_dual_mul_f32 v82, v104, v148 :: v_dual_mul_f32 v83, v115, v149
	v_dual_fmac_f32 v52, v45, v149 :: v_dual_fmac_f32 v53, v54, v148
	v_dual_fmac_f32 v62, v65, v149 :: v_dual_fmac_f32 v63, v74, v148
	v_dual_fmac_f32 v72, v85, v149 :: v_dual_fmac_f32 v73, v94, v148
	v_dual_fmac_f32 v82, v105, v149 :: v_dual_fmac_f32 v83, v114, v148
	v_dual_fmac_f32 v52, v46, v150 :: v_dual_fmac_f32 v53, v56, v150
	v_dual_fmac_f32 v62, v66, v150 :: v_dual_fmac_f32 v63, v76, v150
	v_dual_fmac_f32 v72, v86, v150 :: v_dual_fmac_f32 v73, v96, v150
	v_dual_fmac_f32 v82, v106, v150 :: v_dual_fmac_f32 v83, v116, v150
	v_dual_fmac_f32 v52, v47, v151 :: v_dual_fmac_f32 v53, v57, v151
	v_dual_fmac_f32 v62, v67, v151 :: v_dual_fmac_f32 v63, v77, v151
	v_dual_fmac_f32 v72, v87, v151 :: v_dual_fmac_f32 v73, v97, v151
	v_dual_fmac_f32 v82, v107, v151 :: v_dual_fmac_f32 v83, v117, v151
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v52, v48, v152 :: v_dual_fmac_f32 v53, v58, v152
	v_dual_fmac_f32 v62, v68, v152 :: v_dual_fmac_f32 v63, v78, v152
	v_dual_fmac_f32 v72, v88, v152 :: v_dual_fmac_f32 v73, v98, v152
	v_dual_fmac_f32 v82, v108, v152 :: v_dual_fmac_f32 v83, v118, v152
	v_dual_fmac_f32 v52, v49, v153 :: v_dual_fmac_f32 v53, v59, v153
	v_dual_fmac_f32 v62, v69, v153 :: v_dual_fmac_f32 v63, v79, v153
	v_dual_fmac_f32 v72, v89, v153 :: v_dual_fmac_f32 v73, v99, v153
	v_dual_fmac_f32 v82, v109, v153 :: v_dual_fmac_f32 v83, v119, v153
	v_dual_fmac_f32 v52, v50, v154 :: v_dual_fmac_f32 v53, v60, v154
	v_dual_fmac_f32 v62, v70, v154 :: v_dual_fmac_f32 v63, v80, v154
	v_dual_fmac_f32 v72, v90, v154 :: v_dual_fmac_f32 v73, v100, v154
	v_dual_fmac_f32 v82, v110, v154 :: v_dual_fmac_f32 v83, v120, v154
	v_dual_fmac_f32 v52, v51, v155 :: v_dual_fmac_f32 v53, v61, v155
	v_dual_fmac_f32 v62, v71, v155 :: v_dual_fmac_f32 v63, v81, v155
	v_dual_fmac_f32 v72, v91, v155 :: v_dual_fmac_f32 v73, v101, v155
	v_dual_fmac_f32 v82, v111, v155 :: v_dual_fmac_f32 v83, v121, v155
	global_load_b128 v[148:151], v3, s[28:29]
	global_load_b128 v[152:155], v3, s[28:29] offset:16
	v_dual_add_f32 v140, v52, v140 :: v_dual_add_f32 v141, v141, v53
	v_dual_add_f32 v142, v62, v142 :: v_dual_add_f32 v143, v143, v63
	v_dual_add_f32 v144, v72, v144 :: v_dual_add_f32 v145, v145, v73
	v_dual_add_f32 v146, v82, v146 :: v_dual_add_f32 v147, v147, v83
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s3
	s_wait_loadcnt 0x0
	.Lrx_b1_s3_done:
	v_dual_add_f32 v132, v132, v140 :: v_dual_add_f32 v133, v133, v141
	v_dual_add_f32 v134, v134, v142 :: v_dual_add_f32 v135, v135, v143
	v_dual_add_f32 v136, v136, v144 :: v_dual_add_f32 v137, v137, v145
	v_dual_add_f32 v138, v138, v146 :: v_dual_add_f32 v139, v139, v147
	v_dual_add_f32 v124, v124, v132 :: v_dual_add_f32 v125, v125, v133
	v_dual_add_f32 v126, v126, v134 :: v_dual_add_f32 v127, v127, v135
	v_dual_add_f32 v128, v128, v136 :: v_dual_add_f32 v129, v129, v137
	v_dual_add_f32 v130, v130, v138 :: v_dual_add_f32 v131, v131, v139
	ds_swizzle_b32 v148, v124 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v149, v125 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v150, v126 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v151, v127 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v152, v128 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v153, v129 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v154, v130 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v155, v131 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v148
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v149
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v150
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v151
	s_wait_dscnt 0x3
	v_add_f32_e32 v128, v128, v152
	s_wait_dscnt 0x2
	v_add_f32_e32 v129, v129, v153
	s_wait_dscnt 0x1
	v_add_f32_e32 v130, v130, v154
	s_wait_dscnt 0x0
	v_add_f32_e32 v131, v131, v155
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v148, v3, v124
	ds_bpermute_b32 v149, v3, v125
	ds_bpermute_b32 v150, v3, v126
	ds_bpermute_b32 v151, v3, v127
	ds_bpermute_b32 v152, v3, v128
	ds_bpermute_b32 v153, v3, v129
	ds_bpermute_b32 v154, v3, v130
	ds_bpermute_b32 v155, v3, v131
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v148
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v149
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v150
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v151
	s_wait_dscnt 0x3
	v_add_f32_e32 v128, v128, v152
	s_wait_dscnt 0x2
	v_add_f32_e32 v129, v129, v153
	s_wait_dscnt 0x1
	v_add_f32_e32 v130, v130, v154
	s_wait_dscnt 0x0
	v_add_f32_e32 v131, v131, v155
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v148, v3, v124
	ds_bpermute_b32 v149, v3, v125
	ds_bpermute_b32 v150, v3, v126
	ds_bpermute_b32 v151, v3, v127
	ds_bpermute_b32 v152, v3, v128
	ds_bpermute_b32 v153, v3, v129
	ds_bpermute_b32 v154, v3, v130
	ds_bpermute_b32 v155, v3, v131
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v148
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v149
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v150
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v151
	s_wait_dscnt 0x3
	v_add_f32_e32 v128, v128, v152
	s_wait_dscnt 0x2
	v_add_f32_e32 v129, v129, v153
	s_wait_dscnt 0x1
	v_add_f32_e32 v130, v130, v154
	s_wait_dscnt 0x0
	v_add_f32_e32 v131, v131, v155
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v148, v3, v124
	ds_bpermute_b32 v149, v3, v125
	ds_bpermute_b32 v150, v3, v126
	ds_bpermute_b32 v151, v3, v127
	ds_bpermute_b32 v152, v3, v128
	ds_bpermute_b32 v153, v3, v129
	ds_bpermute_b32 v154, v3, v130
	ds_bpermute_b32 v155, v3, v131
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v148
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v149
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v150
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v151
	s_wait_dscnt 0x3
	v_add_f32_e32 v128, v128, v152
	s_wait_dscnt 0x2
	v_add_f32_e32 v129, v129, v153
	s_wait_dscnt 0x1
	v_add_f32_e32 v130, v130, v154
	s_wait_dscnt 0x0
	v_add_f32_e32 v131, v131, v155
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v148, v3, v124
	ds_bpermute_b32 v149, v3, v125
	ds_bpermute_b32 v150, v3, v126
	ds_bpermute_b32 v151, v3, v127
	ds_bpermute_b32 v152, v3, v128
	ds_bpermute_b32 v153, v3, v129
	ds_bpermute_b32 v154, v3, v130
	ds_bpermute_b32 v155, v3, v131
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v148
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v149
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v150
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v151
	s_wait_dscnt 0x3
	v_add_f32_e32 v128, v128, v152
	s_wait_dscnt 0x2
	v_add_f32_e32 v129, v129, v153
	s_wait_dscnt 0x1
	v_add_f32_e32 v130, v130, v154
	s_wait_dscnt 0x0
	v_add_f32_e32 v131, v131, v155
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	v_mov_b32_e32 v52, s46
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b1_p0_single
	global_load_b32 v44, v52, s[8:9] offset:0
	global_load_b32 v45, v52, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v44, v124, v44
	global_store_b32 v52, v44, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v45, v125, v45
	global_store_b32 v52, v45, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b1_p0_next
	.Lrx_b1_p0_single:
	global_load_b32 v44, v52, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v44, v44, v124
	global_store_b32 v52, v44, s[8:9] offset:0
	s_wait_storecnt 0x0
	s_branch .Lrx_b1_stored
	.Lrx_b1_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b1_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b1_p1_single
	global_load_b32 v46, v52, s[8:9] offset:8
	global_load_b32 v47, v52, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v46, v126, v46
	global_store_b32 v52, v46, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v47, v127, v47
	global_store_b32 v52, v47, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b1_p1_next
	.Lrx_b1_p1_single:
	global_load_b32 v46, v52, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v46, v46, v126
	global_store_b32 v52, v46, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b1_stored
	.Lrx_b1_p1_next:
	s_add_co_i32 s13, s12, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b1_stored
	s_add_co_i32 s13, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b1_p2_single
	global_load_b32 v48, v52, s[8:9] offset:16
	global_load_b32 v49, v52, s[8:9] offset:20
	s_wait_loadcnt 0x1
	v_add_f32_e32 v48, v128, v48
	global_store_b32 v52, v48, s[8:9] offset:16
	s_wait_loadcnt 0x0
	v_add_f32_e32 v49, v129, v49
	global_store_b32 v52, v49, s[8:9] offset:20
	s_wait_storecnt 0x0
	s_branch .Lrx_b1_p2_next
	.Lrx_b1_p2_single:
	global_load_b32 v48, v52, s[8:9] offset:16
	s_wait_loadcnt 0x0
	v_add_f32_e32 v48, v48, v128
	global_store_b32 v52, v48, s[8:9] offset:16
	s_wait_storecnt 0x0
	s_branch .Lrx_b1_stored
	.Lrx_b1_p2_next:
	s_add_co_i32 s13, s12, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b1_stored
	s_add_co_i32 s13, s12, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b1_p3_single
	global_load_b32 v50, v52, s[8:9] offset:24
	global_load_b32 v51, v52, s[8:9] offset:28
	s_wait_loadcnt 0x1
	v_add_f32_e32 v50, v130, v50
	global_store_b32 v52, v50, s[8:9] offset:24
	s_wait_loadcnt 0x0
	v_add_f32_e32 v51, v131, v51
	global_store_b32 v52, v51, s[8:9] offset:28
	s_wait_storecnt 0x0
	s_branch .Lrx_b1_stored
	.Lrx_b1_p3_single:
	global_load_b32 v50, v52, s[8:9] offset:24
	s_wait_loadcnt 0x0
	v_add_f32_e32 v50, v50, v130
	global_store_b32 v52, v50, s[8:9] offset:24
	s_wait_storecnt 0x0
	s_branch .Lrx_b1_stored
	.Lrx_b1_stored:
	s_branch .Lrx_end
	.Lrx_b2:
	s_mul_i32 s12, ttmp9, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v12, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v13, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v14, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v15, s13
	v_add_nc_u32_e32 v7, s13, v1
	s_add_co_i32 s13, s12, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v16, s13
	v_add_nc_u32_e32 v8, s13, v1
	s_add_co_i32 s13, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v17, s13
	v_add_nc_u32_e32 v9, s13, v1
	s_add_co_i32 s13, s12, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v18, s13
	v_add_nc_u32_e32 v10, s13, v1
	s_add_co_i32 s13, s12, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v19, s13
	v_add_nc_u32_e32 v11, s13, v1
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s0_done
	s_mov_b32 s15, 0x0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[172:175], v2, s[28:29]
	global_load_b128 v[176:179], v2, s[28:29] offset:16
	global_load_b128 v[180:183], v2, s[30:31]
	global_load_b128 v[184:187], v2, s[30:31] offset:16
	.Lrx_b2_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0x6
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x4
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v44, v172 :: v_dual_mul_f32 v53, v55, v173
	v_dual_mul_f32 v62, v64, v172 :: v_dual_mul_f32 v63, v75, v173
	v_dual_mul_f32 v72, v84, v172 :: v_dual_mul_f32 v73, v95, v173
	v_dual_mul_f32 v82, v104, v172 :: v_dual_mul_f32 v83, v115, v173
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v92, v44, v180 :: v_dual_mul_f32 v93, v55, v181
	v_dual_mul_f32 v102, v64, v180 :: v_dual_mul_f32 v103, v75, v181
	v_dual_mul_f32 v112, v84, v180 :: v_dual_mul_f32 v113, v95, v181
	v_dual_mul_f32 v122, v104, v180 :: v_dual_mul_f32 v123, v115, v181
	v_dual_fmac_f32 v52, v45, v173 :: v_dual_fmac_f32 v53, v54, v172
	v_dual_fmac_f32 v62, v65, v173 :: v_dual_fmac_f32 v63, v74, v172
	v_dual_fmac_f32 v72, v85, v173 :: v_dual_fmac_f32 v73, v94, v172
	v_dual_fmac_f32 v82, v105, v173 :: v_dual_fmac_f32 v83, v114, v172
	v_dual_fmac_f32 v92, v45, v181 :: v_dual_fmac_f32 v93, v54, v180
	v_dual_fmac_f32 v102, v65, v181 :: v_dual_fmac_f32 v103, v74, v180
	v_dual_fmac_f32 v112, v85, v181 :: v_dual_fmac_f32 v113, v94, v180
	v_dual_fmac_f32 v122, v105, v181 :: v_dual_fmac_f32 v123, v114, v180
	v_dual_fmac_f32 v52, v46, v174 :: v_dual_fmac_f32 v53, v56, v174
	v_dual_fmac_f32 v62, v66, v174 :: v_dual_fmac_f32 v63, v76, v174
	v_dual_fmac_f32 v72, v86, v174 :: v_dual_fmac_f32 v73, v96, v174
	v_dual_fmac_f32 v82, v106, v174 :: v_dual_fmac_f32 v83, v116, v174
	v_dual_fmac_f32 v92, v46, v182 :: v_dual_fmac_f32 v93, v56, v182
	v_dual_fmac_f32 v102, v66, v182 :: v_dual_fmac_f32 v103, v76, v182
	v_dual_fmac_f32 v112, v86, v182 :: v_dual_fmac_f32 v113, v96, v182
	v_dual_fmac_f32 v122, v106, v182 :: v_dual_fmac_f32 v123, v116, v182
	v_dual_fmac_f32 v52, v47, v175 :: v_dual_fmac_f32 v53, v57, v175
	v_dual_fmac_f32 v62, v67, v175 :: v_dual_fmac_f32 v63, v77, v175
	v_dual_fmac_f32 v72, v87, v175 :: v_dual_fmac_f32 v73, v97, v175
	v_dual_fmac_f32 v82, v107, v175 :: v_dual_fmac_f32 v83, v117, v175
	v_dual_fmac_f32 v92, v47, v183 :: v_dual_fmac_f32 v93, v57, v183
	v_dual_fmac_f32 v102, v67, v183 :: v_dual_fmac_f32 v103, v77, v183
	v_dual_fmac_f32 v112, v87, v183 :: v_dual_fmac_f32 v113, v97, v183
	v_dual_fmac_f32 v122, v107, v183 :: v_dual_fmac_f32 v123, v117, v183
	v_dual_fmac_f32 v52, v48, v176 :: v_dual_fmac_f32 v53, v58, v176
	v_dual_fmac_f32 v62, v68, v176 :: v_dual_fmac_f32 v63, v78, v176
	v_dual_fmac_f32 v72, v88, v176 :: v_dual_fmac_f32 v73, v98, v176
	v_dual_fmac_f32 v82, v108, v176 :: v_dual_fmac_f32 v83, v118, v176
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v92, v48, v184 :: v_dual_fmac_f32 v93, v58, v184
	v_dual_fmac_f32 v102, v68, v184 :: v_dual_fmac_f32 v103, v78, v184
	v_dual_fmac_f32 v112, v88, v184 :: v_dual_fmac_f32 v113, v98, v184
	v_dual_fmac_f32 v122, v108, v184 :: v_dual_fmac_f32 v123, v118, v184
	v_dual_fmac_f32 v52, v49, v177 :: v_dual_fmac_f32 v53, v59, v177
	v_dual_fmac_f32 v62, v69, v177 :: v_dual_fmac_f32 v63, v79, v177
	v_dual_fmac_f32 v72, v89, v177 :: v_dual_fmac_f32 v73, v99, v177
	v_dual_fmac_f32 v82, v109, v177 :: v_dual_fmac_f32 v83, v119, v177
	v_dual_fmac_f32 v92, v49, v185 :: v_dual_fmac_f32 v93, v59, v185
	v_dual_fmac_f32 v102, v69, v185 :: v_dual_fmac_f32 v103, v79, v185
	v_dual_fmac_f32 v112, v89, v185 :: v_dual_fmac_f32 v113, v99, v185
	v_dual_fmac_f32 v122, v109, v185 :: v_dual_fmac_f32 v123, v119, v185
	v_dual_fmac_f32 v52, v50, v178 :: v_dual_fmac_f32 v53, v60, v178
	v_dual_fmac_f32 v62, v70, v178 :: v_dual_fmac_f32 v63, v80, v178
	v_dual_fmac_f32 v72, v90, v178 :: v_dual_fmac_f32 v73, v100, v178
	v_dual_fmac_f32 v82, v110, v178 :: v_dual_fmac_f32 v83, v120, v178
	v_dual_fmac_f32 v92, v50, v186 :: v_dual_fmac_f32 v93, v60, v186
	v_dual_fmac_f32 v102, v70, v186 :: v_dual_fmac_f32 v103, v80, v186
	v_dual_fmac_f32 v112, v90, v186 :: v_dual_fmac_f32 v113, v100, v186
	v_dual_fmac_f32 v122, v110, v186 :: v_dual_fmac_f32 v123, v120, v186
	v_dual_fmac_f32 v52, v51, v179 :: v_dual_fmac_f32 v53, v61, v179
	v_dual_fmac_f32 v62, v71, v179 :: v_dual_fmac_f32 v63, v81, v179
	v_dual_fmac_f32 v72, v91, v179 :: v_dual_fmac_f32 v73, v101, v179
	v_dual_fmac_f32 v82, v111, v179 :: v_dual_fmac_f32 v83, v121, v179
	v_dual_fmac_f32 v92, v51, v187 :: v_dual_fmac_f32 v93, v61, v187
	v_dual_fmac_f32 v102, v71, v187 :: v_dual_fmac_f32 v103, v81, v187
	v_dual_fmac_f32 v112, v91, v187 :: v_dual_fmac_f32 v113, v101, v187
	v_dual_fmac_f32 v122, v111, v187 :: v_dual_fmac_f32 v123, v121, v187
	global_load_b128 v[172:175], v3, s[28:29]
	global_load_b128 v[176:179], v3, s[28:29] offset:16
	global_load_b128 v[180:183], v3, s[30:31]
	global_load_b128 v[184:187], v3, s[30:31] offset:16
	v_dual_add_f32 v124, v52, v124 :: v_dual_add_f32 v125, v125, v53
	v_dual_add_f32 v126, v62, v126 :: v_dual_add_f32 v127, v127, v63
	v_dual_add_f32 v128, v72, v128 :: v_dual_add_f32 v129, v129, v73
	v_dual_add_f32 v130, v82, v130 :: v_dual_add_f32 v131, v131, v83
	v_dual_add_f32 v148, v92, v148 :: v_dual_add_f32 v149, v149, v93
	v_dual_add_f32 v150, v102, v150 :: v_dual_add_f32 v151, v151, v103
	v_dual_add_f32 v152, v112, v152 :: v_dual_add_f32 v153, v153, v113
	v_dual_add_f32 v154, v122, v154 :: v_dual_add_f32 v155, v155, v123
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s0
	s_wait_loadcnt 0x0
	.Lrx_b2_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[172:175], v2, s[28:29]
	global_load_b128 v[176:179], v2, s[28:29] offset:16
	global_load_b128 v[180:183], v2, s[30:31]
	global_load_b128 v[184:187], v2, s[30:31] offset:16
	.Lrx_b2_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0x6
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x4
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v44, v172 :: v_dual_mul_f32 v53, v55, v173
	v_dual_mul_f32 v62, v64, v172 :: v_dual_mul_f32 v63, v75, v173
	v_dual_mul_f32 v72, v84, v172 :: v_dual_mul_f32 v73, v95, v173
	v_dual_mul_f32 v82, v104, v172 :: v_dual_mul_f32 v83, v115, v173
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v92, v44, v180 :: v_dual_mul_f32 v93, v55, v181
	v_dual_mul_f32 v102, v64, v180 :: v_dual_mul_f32 v103, v75, v181
	v_dual_mul_f32 v112, v84, v180 :: v_dual_mul_f32 v113, v95, v181
	v_dual_mul_f32 v122, v104, v180 :: v_dual_mul_f32 v123, v115, v181
	v_dual_fmac_f32 v52, v45, v173 :: v_dual_fmac_f32 v53, v54, v172
	v_dual_fmac_f32 v62, v65, v173 :: v_dual_fmac_f32 v63, v74, v172
	v_dual_fmac_f32 v72, v85, v173 :: v_dual_fmac_f32 v73, v94, v172
	v_dual_fmac_f32 v82, v105, v173 :: v_dual_fmac_f32 v83, v114, v172
	v_dual_fmac_f32 v92, v45, v181 :: v_dual_fmac_f32 v93, v54, v180
	v_dual_fmac_f32 v102, v65, v181 :: v_dual_fmac_f32 v103, v74, v180
	v_dual_fmac_f32 v112, v85, v181 :: v_dual_fmac_f32 v113, v94, v180
	v_dual_fmac_f32 v122, v105, v181 :: v_dual_fmac_f32 v123, v114, v180
	v_dual_fmac_f32 v52, v46, v174 :: v_dual_fmac_f32 v53, v56, v174
	v_dual_fmac_f32 v62, v66, v174 :: v_dual_fmac_f32 v63, v76, v174
	v_dual_fmac_f32 v72, v86, v174 :: v_dual_fmac_f32 v73, v96, v174
	v_dual_fmac_f32 v82, v106, v174 :: v_dual_fmac_f32 v83, v116, v174
	v_dual_fmac_f32 v92, v46, v182 :: v_dual_fmac_f32 v93, v56, v182
	v_dual_fmac_f32 v102, v66, v182 :: v_dual_fmac_f32 v103, v76, v182
	v_dual_fmac_f32 v112, v86, v182 :: v_dual_fmac_f32 v113, v96, v182
	v_dual_fmac_f32 v122, v106, v182 :: v_dual_fmac_f32 v123, v116, v182
	v_dual_fmac_f32 v52, v47, v175 :: v_dual_fmac_f32 v53, v57, v175
	v_dual_fmac_f32 v62, v67, v175 :: v_dual_fmac_f32 v63, v77, v175
	v_dual_fmac_f32 v72, v87, v175 :: v_dual_fmac_f32 v73, v97, v175
	v_dual_fmac_f32 v82, v107, v175 :: v_dual_fmac_f32 v83, v117, v175
	v_dual_fmac_f32 v92, v47, v183 :: v_dual_fmac_f32 v93, v57, v183
	v_dual_fmac_f32 v102, v67, v183 :: v_dual_fmac_f32 v103, v77, v183
	v_dual_fmac_f32 v112, v87, v183 :: v_dual_fmac_f32 v113, v97, v183
	v_dual_fmac_f32 v122, v107, v183 :: v_dual_fmac_f32 v123, v117, v183
	v_dual_fmac_f32 v52, v48, v176 :: v_dual_fmac_f32 v53, v58, v176
	v_dual_fmac_f32 v62, v68, v176 :: v_dual_fmac_f32 v63, v78, v176
	v_dual_fmac_f32 v72, v88, v176 :: v_dual_fmac_f32 v73, v98, v176
	v_dual_fmac_f32 v82, v108, v176 :: v_dual_fmac_f32 v83, v118, v176
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v92, v48, v184 :: v_dual_fmac_f32 v93, v58, v184
	v_dual_fmac_f32 v102, v68, v184 :: v_dual_fmac_f32 v103, v78, v184
	v_dual_fmac_f32 v112, v88, v184 :: v_dual_fmac_f32 v113, v98, v184
	v_dual_fmac_f32 v122, v108, v184 :: v_dual_fmac_f32 v123, v118, v184
	v_dual_fmac_f32 v52, v49, v177 :: v_dual_fmac_f32 v53, v59, v177
	v_dual_fmac_f32 v62, v69, v177 :: v_dual_fmac_f32 v63, v79, v177
	v_dual_fmac_f32 v72, v89, v177 :: v_dual_fmac_f32 v73, v99, v177
	v_dual_fmac_f32 v82, v109, v177 :: v_dual_fmac_f32 v83, v119, v177
	v_dual_fmac_f32 v92, v49, v185 :: v_dual_fmac_f32 v93, v59, v185
	v_dual_fmac_f32 v102, v69, v185 :: v_dual_fmac_f32 v103, v79, v185
	v_dual_fmac_f32 v112, v89, v185 :: v_dual_fmac_f32 v113, v99, v185
	v_dual_fmac_f32 v122, v109, v185 :: v_dual_fmac_f32 v123, v119, v185
	v_dual_fmac_f32 v52, v50, v178 :: v_dual_fmac_f32 v53, v60, v178
	v_dual_fmac_f32 v62, v70, v178 :: v_dual_fmac_f32 v63, v80, v178
	v_dual_fmac_f32 v72, v90, v178 :: v_dual_fmac_f32 v73, v100, v178
	v_dual_fmac_f32 v82, v110, v178 :: v_dual_fmac_f32 v83, v120, v178
	v_dual_fmac_f32 v92, v50, v186 :: v_dual_fmac_f32 v93, v60, v186
	v_dual_fmac_f32 v102, v70, v186 :: v_dual_fmac_f32 v103, v80, v186
	v_dual_fmac_f32 v112, v90, v186 :: v_dual_fmac_f32 v113, v100, v186
	v_dual_fmac_f32 v122, v110, v186 :: v_dual_fmac_f32 v123, v120, v186
	v_dual_fmac_f32 v52, v51, v179 :: v_dual_fmac_f32 v53, v61, v179
	v_dual_fmac_f32 v62, v71, v179 :: v_dual_fmac_f32 v63, v81, v179
	v_dual_fmac_f32 v72, v91, v179 :: v_dual_fmac_f32 v73, v101, v179
	v_dual_fmac_f32 v82, v111, v179 :: v_dual_fmac_f32 v83, v121, v179
	v_dual_fmac_f32 v92, v51, v187 :: v_dual_fmac_f32 v93, v61, v187
	v_dual_fmac_f32 v102, v71, v187 :: v_dual_fmac_f32 v103, v81, v187
	v_dual_fmac_f32 v112, v91, v187 :: v_dual_fmac_f32 v113, v101, v187
	v_dual_fmac_f32 v122, v111, v187 :: v_dual_fmac_f32 v123, v121, v187
	global_load_b128 v[172:175], v3, s[28:29]
	global_load_b128 v[176:179], v3, s[28:29] offset:16
	global_load_b128 v[180:183], v3, s[30:31]
	global_load_b128 v[184:187], v3, s[30:31] offset:16
	v_dual_add_f32 v132, v52, v132 :: v_dual_add_f32 v133, v133, v53
	v_dual_add_f32 v134, v62, v134 :: v_dual_add_f32 v135, v135, v63
	v_dual_add_f32 v136, v72, v136 :: v_dual_add_f32 v137, v137, v73
	v_dual_add_f32 v138, v82, v138 :: v_dual_add_f32 v139, v139, v83
	v_dual_add_f32 v156, v92, v156 :: v_dual_add_f32 v157, v157, v93
	v_dual_add_f32 v158, v102, v158 :: v_dual_add_f32 v159, v159, v103
	v_dual_add_f32 v160, v112, v160 :: v_dual_add_f32 v161, v161, v113
	v_dual_add_f32 v162, v122, v162 :: v_dual_add_f32 v163, v163, v123
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s1
	s_wait_loadcnt 0x0
	.Lrx_b2_s1_done:
	v_dual_add_f32 v124, v124, v132 :: v_dual_add_f32 v125, v125, v133
	v_dual_add_f32 v126, v126, v134 :: v_dual_add_f32 v127, v127, v135
	v_dual_add_f32 v128, v128, v136 :: v_dual_add_f32 v129, v129, v137
	v_dual_add_f32 v130, v130, v138 :: v_dual_add_f32 v131, v131, v139
	v_dual_add_f32 v148, v148, v156 :: v_dual_add_f32 v149, v149, v157
	v_dual_add_f32 v150, v150, v158 :: v_dual_add_f32 v151, v151, v159
	v_dual_add_f32 v152, v152, v160 :: v_dual_add_f32 v153, v153, v161
	v_dual_add_f32 v154, v154, v162 :: v_dual_add_f32 v155, v155, v163
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[172:175], v2, s[28:29]
	global_load_b128 v[176:179], v2, s[28:29] offset:16
	global_load_b128 v[180:183], v2, s[30:31]
	global_load_b128 v[184:187], v2, s[30:31] offset:16
	.Lrx_b2_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0x6
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x4
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v44, v172 :: v_dual_mul_f32 v53, v55, v173
	v_dual_mul_f32 v62, v64, v172 :: v_dual_mul_f32 v63, v75, v173
	v_dual_mul_f32 v72, v84, v172 :: v_dual_mul_f32 v73, v95, v173
	v_dual_mul_f32 v82, v104, v172 :: v_dual_mul_f32 v83, v115, v173
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v92, v44, v180 :: v_dual_mul_f32 v93, v55, v181
	v_dual_mul_f32 v102, v64, v180 :: v_dual_mul_f32 v103, v75, v181
	v_dual_mul_f32 v112, v84, v180 :: v_dual_mul_f32 v113, v95, v181
	v_dual_mul_f32 v122, v104, v180 :: v_dual_mul_f32 v123, v115, v181
	v_dual_fmac_f32 v52, v45, v173 :: v_dual_fmac_f32 v53, v54, v172
	v_dual_fmac_f32 v62, v65, v173 :: v_dual_fmac_f32 v63, v74, v172
	v_dual_fmac_f32 v72, v85, v173 :: v_dual_fmac_f32 v73, v94, v172
	v_dual_fmac_f32 v82, v105, v173 :: v_dual_fmac_f32 v83, v114, v172
	v_dual_fmac_f32 v92, v45, v181 :: v_dual_fmac_f32 v93, v54, v180
	v_dual_fmac_f32 v102, v65, v181 :: v_dual_fmac_f32 v103, v74, v180
	v_dual_fmac_f32 v112, v85, v181 :: v_dual_fmac_f32 v113, v94, v180
	v_dual_fmac_f32 v122, v105, v181 :: v_dual_fmac_f32 v123, v114, v180
	v_dual_fmac_f32 v52, v46, v174 :: v_dual_fmac_f32 v53, v56, v174
	v_dual_fmac_f32 v62, v66, v174 :: v_dual_fmac_f32 v63, v76, v174
	v_dual_fmac_f32 v72, v86, v174 :: v_dual_fmac_f32 v73, v96, v174
	v_dual_fmac_f32 v82, v106, v174 :: v_dual_fmac_f32 v83, v116, v174
	v_dual_fmac_f32 v92, v46, v182 :: v_dual_fmac_f32 v93, v56, v182
	v_dual_fmac_f32 v102, v66, v182 :: v_dual_fmac_f32 v103, v76, v182
	v_dual_fmac_f32 v112, v86, v182 :: v_dual_fmac_f32 v113, v96, v182
	v_dual_fmac_f32 v122, v106, v182 :: v_dual_fmac_f32 v123, v116, v182
	v_dual_fmac_f32 v52, v47, v175 :: v_dual_fmac_f32 v53, v57, v175
	v_dual_fmac_f32 v62, v67, v175 :: v_dual_fmac_f32 v63, v77, v175
	v_dual_fmac_f32 v72, v87, v175 :: v_dual_fmac_f32 v73, v97, v175
	v_dual_fmac_f32 v82, v107, v175 :: v_dual_fmac_f32 v83, v117, v175
	v_dual_fmac_f32 v92, v47, v183 :: v_dual_fmac_f32 v93, v57, v183
	v_dual_fmac_f32 v102, v67, v183 :: v_dual_fmac_f32 v103, v77, v183
	v_dual_fmac_f32 v112, v87, v183 :: v_dual_fmac_f32 v113, v97, v183
	v_dual_fmac_f32 v122, v107, v183 :: v_dual_fmac_f32 v123, v117, v183
	v_dual_fmac_f32 v52, v48, v176 :: v_dual_fmac_f32 v53, v58, v176
	v_dual_fmac_f32 v62, v68, v176 :: v_dual_fmac_f32 v63, v78, v176
	v_dual_fmac_f32 v72, v88, v176 :: v_dual_fmac_f32 v73, v98, v176
	v_dual_fmac_f32 v82, v108, v176 :: v_dual_fmac_f32 v83, v118, v176
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v92, v48, v184 :: v_dual_fmac_f32 v93, v58, v184
	v_dual_fmac_f32 v102, v68, v184 :: v_dual_fmac_f32 v103, v78, v184
	v_dual_fmac_f32 v112, v88, v184 :: v_dual_fmac_f32 v113, v98, v184
	v_dual_fmac_f32 v122, v108, v184 :: v_dual_fmac_f32 v123, v118, v184
	v_dual_fmac_f32 v52, v49, v177 :: v_dual_fmac_f32 v53, v59, v177
	v_dual_fmac_f32 v62, v69, v177 :: v_dual_fmac_f32 v63, v79, v177
	v_dual_fmac_f32 v72, v89, v177 :: v_dual_fmac_f32 v73, v99, v177
	v_dual_fmac_f32 v82, v109, v177 :: v_dual_fmac_f32 v83, v119, v177
	v_dual_fmac_f32 v92, v49, v185 :: v_dual_fmac_f32 v93, v59, v185
	v_dual_fmac_f32 v102, v69, v185 :: v_dual_fmac_f32 v103, v79, v185
	v_dual_fmac_f32 v112, v89, v185 :: v_dual_fmac_f32 v113, v99, v185
	v_dual_fmac_f32 v122, v109, v185 :: v_dual_fmac_f32 v123, v119, v185
	v_dual_fmac_f32 v52, v50, v178 :: v_dual_fmac_f32 v53, v60, v178
	v_dual_fmac_f32 v62, v70, v178 :: v_dual_fmac_f32 v63, v80, v178
	v_dual_fmac_f32 v72, v90, v178 :: v_dual_fmac_f32 v73, v100, v178
	v_dual_fmac_f32 v82, v110, v178 :: v_dual_fmac_f32 v83, v120, v178
	v_dual_fmac_f32 v92, v50, v186 :: v_dual_fmac_f32 v93, v60, v186
	v_dual_fmac_f32 v102, v70, v186 :: v_dual_fmac_f32 v103, v80, v186
	v_dual_fmac_f32 v112, v90, v186 :: v_dual_fmac_f32 v113, v100, v186
	v_dual_fmac_f32 v122, v110, v186 :: v_dual_fmac_f32 v123, v120, v186
	v_dual_fmac_f32 v52, v51, v179 :: v_dual_fmac_f32 v53, v61, v179
	v_dual_fmac_f32 v62, v71, v179 :: v_dual_fmac_f32 v63, v81, v179
	v_dual_fmac_f32 v72, v91, v179 :: v_dual_fmac_f32 v73, v101, v179
	v_dual_fmac_f32 v82, v111, v179 :: v_dual_fmac_f32 v83, v121, v179
	v_dual_fmac_f32 v92, v51, v187 :: v_dual_fmac_f32 v93, v61, v187
	v_dual_fmac_f32 v102, v71, v187 :: v_dual_fmac_f32 v103, v81, v187
	v_dual_fmac_f32 v112, v91, v187 :: v_dual_fmac_f32 v113, v101, v187
	v_dual_fmac_f32 v122, v111, v187 :: v_dual_fmac_f32 v123, v121, v187
	global_load_b128 v[172:175], v3, s[28:29]
	global_load_b128 v[176:179], v3, s[28:29] offset:16
	global_load_b128 v[180:183], v3, s[30:31]
	global_load_b128 v[184:187], v3, s[30:31] offset:16
	v_dual_add_f32 v132, v52, v132 :: v_dual_add_f32 v133, v133, v53
	v_dual_add_f32 v134, v62, v134 :: v_dual_add_f32 v135, v135, v63
	v_dual_add_f32 v136, v72, v136 :: v_dual_add_f32 v137, v137, v73
	v_dual_add_f32 v138, v82, v138 :: v_dual_add_f32 v139, v139, v83
	v_dual_add_f32 v156, v92, v156 :: v_dual_add_f32 v157, v157, v93
	v_dual_add_f32 v158, v102, v158 :: v_dual_add_f32 v159, v159, v103
	v_dual_add_f32 v160, v112, v160 :: v_dual_add_f32 v161, v161, v113
	v_dual_add_f32 v162, v122, v162 :: v_dual_add_f32 v163, v163, v123
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s2
	s_wait_loadcnt 0x0
	.Lrx_b2_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[172:175], v2, s[28:29]
	global_load_b128 v[176:179], v2, s[28:29] offset:16
	global_load_b128 v[180:183], v2, s[30:31]
	global_load_b128 v[184:187], v2, s[30:31] offset:16
	.Lrx_b2_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0x6
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x4
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v44, v172 :: v_dual_mul_f32 v53, v55, v173
	v_dual_mul_f32 v62, v64, v172 :: v_dual_mul_f32 v63, v75, v173
	v_dual_mul_f32 v72, v84, v172 :: v_dual_mul_f32 v73, v95, v173
	v_dual_mul_f32 v82, v104, v172 :: v_dual_mul_f32 v83, v115, v173
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v92, v44, v180 :: v_dual_mul_f32 v93, v55, v181
	v_dual_mul_f32 v102, v64, v180 :: v_dual_mul_f32 v103, v75, v181
	v_dual_mul_f32 v112, v84, v180 :: v_dual_mul_f32 v113, v95, v181
	v_dual_mul_f32 v122, v104, v180 :: v_dual_mul_f32 v123, v115, v181
	v_dual_fmac_f32 v52, v45, v173 :: v_dual_fmac_f32 v53, v54, v172
	v_dual_fmac_f32 v62, v65, v173 :: v_dual_fmac_f32 v63, v74, v172
	v_dual_fmac_f32 v72, v85, v173 :: v_dual_fmac_f32 v73, v94, v172
	v_dual_fmac_f32 v82, v105, v173 :: v_dual_fmac_f32 v83, v114, v172
	v_dual_fmac_f32 v92, v45, v181 :: v_dual_fmac_f32 v93, v54, v180
	v_dual_fmac_f32 v102, v65, v181 :: v_dual_fmac_f32 v103, v74, v180
	v_dual_fmac_f32 v112, v85, v181 :: v_dual_fmac_f32 v113, v94, v180
	v_dual_fmac_f32 v122, v105, v181 :: v_dual_fmac_f32 v123, v114, v180
	v_dual_fmac_f32 v52, v46, v174 :: v_dual_fmac_f32 v53, v56, v174
	v_dual_fmac_f32 v62, v66, v174 :: v_dual_fmac_f32 v63, v76, v174
	v_dual_fmac_f32 v72, v86, v174 :: v_dual_fmac_f32 v73, v96, v174
	v_dual_fmac_f32 v82, v106, v174 :: v_dual_fmac_f32 v83, v116, v174
	v_dual_fmac_f32 v92, v46, v182 :: v_dual_fmac_f32 v93, v56, v182
	v_dual_fmac_f32 v102, v66, v182 :: v_dual_fmac_f32 v103, v76, v182
	v_dual_fmac_f32 v112, v86, v182 :: v_dual_fmac_f32 v113, v96, v182
	v_dual_fmac_f32 v122, v106, v182 :: v_dual_fmac_f32 v123, v116, v182
	v_dual_fmac_f32 v52, v47, v175 :: v_dual_fmac_f32 v53, v57, v175
	v_dual_fmac_f32 v62, v67, v175 :: v_dual_fmac_f32 v63, v77, v175
	v_dual_fmac_f32 v72, v87, v175 :: v_dual_fmac_f32 v73, v97, v175
	v_dual_fmac_f32 v82, v107, v175 :: v_dual_fmac_f32 v83, v117, v175
	v_dual_fmac_f32 v92, v47, v183 :: v_dual_fmac_f32 v93, v57, v183
	v_dual_fmac_f32 v102, v67, v183 :: v_dual_fmac_f32 v103, v77, v183
	v_dual_fmac_f32 v112, v87, v183 :: v_dual_fmac_f32 v113, v97, v183
	v_dual_fmac_f32 v122, v107, v183 :: v_dual_fmac_f32 v123, v117, v183
	v_dual_fmac_f32 v52, v48, v176 :: v_dual_fmac_f32 v53, v58, v176
	v_dual_fmac_f32 v62, v68, v176 :: v_dual_fmac_f32 v63, v78, v176
	v_dual_fmac_f32 v72, v88, v176 :: v_dual_fmac_f32 v73, v98, v176
	v_dual_fmac_f32 v82, v108, v176 :: v_dual_fmac_f32 v83, v118, v176
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v92, v48, v184 :: v_dual_fmac_f32 v93, v58, v184
	v_dual_fmac_f32 v102, v68, v184 :: v_dual_fmac_f32 v103, v78, v184
	v_dual_fmac_f32 v112, v88, v184 :: v_dual_fmac_f32 v113, v98, v184
	v_dual_fmac_f32 v122, v108, v184 :: v_dual_fmac_f32 v123, v118, v184
	v_dual_fmac_f32 v52, v49, v177 :: v_dual_fmac_f32 v53, v59, v177
	v_dual_fmac_f32 v62, v69, v177 :: v_dual_fmac_f32 v63, v79, v177
	v_dual_fmac_f32 v72, v89, v177 :: v_dual_fmac_f32 v73, v99, v177
	v_dual_fmac_f32 v82, v109, v177 :: v_dual_fmac_f32 v83, v119, v177
	v_dual_fmac_f32 v92, v49, v185 :: v_dual_fmac_f32 v93, v59, v185
	v_dual_fmac_f32 v102, v69, v185 :: v_dual_fmac_f32 v103, v79, v185
	v_dual_fmac_f32 v112, v89, v185 :: v_dual_fmac_f32 v113, v99, v185
	v_dual_fmac_f32 v122, v109, v185 :: v_dual_fmac_f32 v123, v119, v185
	v_dual_fmac_f32 v52, v50, v178 :: v_dual_fmac_f32 v53, v60, v178
	v_dual_fmac_f32 v62, v70, v178 :: v_dual_fmac_f32 v63, v80, v178
	v_dual_fmac_f32 v72, v90, v178 :: v_dual_fmac_f32 v73, v100, v178
	v_dual_fmac_f32 v82, v110, v178 :: v_dual_fmac_f32 v83, v120, v178
	v_dual_fmac_f32 v92, v50, v186 :: v_dual_fmac_f32 v93, v60, v186
	v_dual_fmac_f32 v102, v70, v186 :: v_dual_fmac_f32 v103, v80, v186
	v_dual_fmac_f32 v112, v90, v186 :: v_dual_fmac_f32 v113, v100, v186
	v_dual_fmac_f32 v122, v110, v186 :: v_dual_fmac_f32 v123, v120, v186
	v_dual_fmac_f32 v52, v51, v179 :: v_dual_fmac_f32 v53, v61, v179
	v_dual_fmac_f32 v62, v71, v179 :: v_dual_fmac_f32 v63, v81, v179
	v_dual_fmac_f32 v72, v91, v179 :: v_dual_fmac_f32 v73, v101, v179
	v_dual_fmac_f32 v82, v111, v179 :: v_dual_fmac_f32 v83, v121, v179
	v_dual_fmac_f32 v92, v51, v187 :: v_dual_fmac_f32 v93, v61, v187
	v_dual_fmac_f32 v102, v71, v187 :: v_dual_fmac_f32 v103, v81, v187
	v_dual_fmac_f32 v112, v91, v187 :: v_dual_fmac_f32 v113, v101, v187
	v_dual_fmac_f32 v122, v111, v187 :: v_dual_fmac_f32 v123, v121, v187
	global_load_b128 v[172:175], v3, s[28:29]
	global_load_b128 v[176:179], v3, s[28:29] offset:16
	global_load_b128 v[180:183], v3, s[30:31]
	global_load_b128 v[184:187], v3, s[30:31] offset:16
	v_dual_add_f32 v140, v52, v140 :: v_dual_add_f32 v141, v141, v53
	v_dual_add_f32 v142, v62, v142 :: v_dual_add_f32 v143, v143, v63
	v_dual_add_f32 v144, v72, v144 :: v_dual_add_f32 v145, v145, v73
	v_dual_add_f32 v146, v82, v146 :: v_dual_add_f32 v147, v147, v83
	v_dual_add_f32 v164, v92, v164 :: v_dual_add_f32 v165, v165, v93
	v_dual_add_f32 v166, v102, v166 :: v_dual_add_f32 v167, v167, v103
	v_dual_add_f32 v168, v112, v168 :: v_dual_add_f32 v169, v169, v113
	v_dual_add_f32 v170, v122, v170 :: v_dual_add_f32 v171, v171, v123
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s3
	s_wait_loadcnt 0x0
	.Lrx_b2_s3_done:
	v_dual_add_f32 v132, v132, v140 :: v_dual_add_f32 v133, v133, v141
	v_dual_add_f32 v134, v134, v142 :: v_dual_add_f32 v135, v135, v143
	v_dual_add_f32 v136, v136, v144 :: v_dual_add_f32 v137, v137, v145
	v_dual_add_f32 v138, v138, v146 :: v_dual_add_f32 v139, v139, v147
	v_dual_add_f32 v156, v156, v164 :: v_dual_add_f32 v157, v157, v165
	v_dual_add_f32 v158, v158, v166 :: v_dual_add_f32 v159, v159, v167
	v_dual_add_f32 v160, v160, v168 :: v_dual_add_f32 v161, v161, v169
	v_dual_add_f32 v162, v162, v170 :: v_dual_add_f32 v163, v163, v171
	v_dual_add_f32 v124, v124, v132 :: v_dual_add_f32 v125, v125, v133
	v_dual_add_f32 v126, v126, v134 :: v_dual_add_f32 v127, v127, v135
	v_dual_add_f32 v128, v128, v136 :: v_dual_add_f32 v129, v129, v137
	v_dual_add_f32 v130, v130, v138 :: v_dual_add_f32 v131, v131, v139
	v_dual_add_f32 v148, v148, v156 :: v_dual_add_f32 v149, v149, v157
	v_dual_add_f32 v150, v150, v158 :: v_dual_add_f32 v151, v151, v159
	v_dual_add_f32 v152, v152, v160 :: v_dual_add_f32 v153, v153, v161
	v_dual_add_f32 v154, v154, v162 :: v_dual_add_f32 v155, v155, v163
	ds_swizzle_b32 v172, v124 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v173, v125 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v174, v126 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v175, v127 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v176, v128 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v177, v129 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v178, v130 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v179, v131 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v180, v148 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v181, v149 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v182, v150 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v183, v151 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v184, v152 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v185, v153 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v186, v154 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v187, v155 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0xf
	v_add_f32_e32 v124, v124, v172
	s_wait_dscnt 0xe
	v_add_f32_e32 v125, v125, v173
	s_wait_dscnt 0xd
	v_add_f32_e32 v126, v126, v174
	s_wait_dscnt 0xc
	v_add_f32_e32 v127, v127, v175
	s_wait_dscnt 0xb
	v_add_f32_e32 v128, v128, v176
	s_wait_dscnt 0xa
	v_add_f32_e32 v129, v129, v177
	s_wait_dscnt 0x9
	v_add_f32_e32 v130, v130, v178
	s_wait_dscnt 0x8
	v_add_f32_e32 v131, v131, v179
	s_wait_dscnt 0x7
	v_add_f32_e32 v148, v148, v180
	s_wait_dscnt 0x6
	v_add_f32_e32 v149, v149, v181
	s_wait_dscnt 0x5
	v_add_f32_e32 v150, v150, v182
	s_wait_dscnt 0x4
	v_add_f32_e32 v151, v151, v183
	s_wait_dscnt 0x3
	v_add_f32_e32 v152, v152, v184
	s_wait_dscnt 0x2
	v_add_f32_e32 v153, v153, v185
	s_wait_dscnt 0x1
	v_add_f32_e32 v154, v154, v186
	s_wait_dscnt 0x0
	v_add_f32_e32 v155, v155, v187
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v172, v3, v124
	ds_bpermute_b32 v173, v3, v125
	ds_bpermute_b32 v174, v3, v126
	ds_bpermute_b32 v175, v3, v127
	ds_bpermute_b32 v176, v3, v128
	ds_bpermute_b32 v177, v3, v129
	ds_bpermute_b32 v178, v3, v130
	ds_bpermute_b32 v179, v3, v131
	ds_bpermute_b32 v180, v3, v148
	ds_bpermute_b32 v181, v3, v149
	ds_bpermute_b32 v182, v3, v150
	ds_bpermute_b32 v183, v3, v151
	ds_bpermute_b32 v184, v3, v152
	ds_bpermute_b32 v185, v3, v153
	ds_bpermute_b32 v186, v3, v154
	ds_bpermute_b32 v187, v3, v155
	s_wait_dscnt 0xf
	v_add_f32_e32 v124, v124, v172
	s_wait_dscnt 0xe
	v_add_f32_e32 v125, v125, v173
	s_wait_dscnt 0xd
	v_add_f32_e32 v126, v126, v174
	s_wait_dscnt 0xc
	v_add_f32_e32 v127, v127, v175
	s_wait_dscnt 0xb
	v_add_f32_e32 v128, v128, v176
	s_wait_dscnt 0xa
	v_add_f32_e32 v129, v129, v177
	s_wait_dscnt 0x9
	v_add_f32_e32 v130, v130, v178
	s_wait_dscnt 0x8
	v_add_f32_e32 v131, v131, v179
	s_wait_dscnt 0x7
	v_add_f32_e32 v148, v148, v180
	s_wait_dscnt 0x6
	v_add_f32_e32 v149, v149, v181
	s_wait_dscnt 0x5
	v_add_f32_e32 v150, v150, v182
	s_wait_dscnt 0x4
	v_add_f32_e32 v151, v151, v183
	s_wait_dscnt 0x3
	v_add_f32_e32 v152, v152, v184
	s_wait_dscnt 0x2
	v_add_f32_e32 v153, v153, v185
	s_wait_dscnt 0x1
	v_add_f32_e32 v154, v154, v186
	s_wait_dscnt 0x0
	v_add_f32_e32 v155, v155, v187
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v172, v3, v124
	ds_bpermute_b32 v173, v3, v125
	ds_bpermute_b32 v174, v3, v126
	ds_bpermute_b32 v175, v3, v127
	ds_bpermute_b32 v176, v3, v128
	ds_bpermute_b32 v177, v3, v129
	ds_bpermute_b32 v178, v3, v130
	ds_bpermute_b32 v179, v3, v131
	ds_bpermute_b32 v180, v3, v148
	ds_bpermute_b32 v181, v3, v149
	ds_bpermute_b32 v182, v3, v150
	ds_bpermute_b32 v183, v3, v151
	ds_bpermute_b32 v184, v3, v152
	ds_bpermute_b32 v185, v3, v153
	ds_bpermute_b32 v186, v3, v154
	ds_bpermute_b32 v187, v3, v155
	s_wait_dscnt 0xf
	v_add_f32_e32 v124, v124, v172
	s_wait_dscnt 0xe
	v_add_f32_e32 v125, v125, v173
	s_wait_dscnt 0xd
	v_add_f32_e32 v126, v126, v174
	s_wait_dscnt 0xc
	v_add_f32_e32 v127, v127, v175
	s_wait_dscnt 0xb
	v_add_f32_e32 v128, v128, v176
	s_wait_dscnt 0xa
	v_add_f32_e32 v129, v129, v177
	s_wait_dscnt 0x9
	v_add_f32_e32 v130, v130, v178
	s_wait_dscnt 0x8
	v_add_f32_e32 v131, v131, v179
	s_wait_dscnt 0x7
	v_add_f32_e32 v148, v148, v180
	s_wait_dscnt 0x6
	v_add_f32_e32 v149, v149, v181
	s_wait_dscnt 0x5
	v_add_f32_e32 v150, v150, v182
	s_wait_dscnt 0x4
	v_add_f32_e32 v151, v151, v183
	s_wait_dscnt 0x3
	v_add_f32_e32 v152, v152, v184
	s_wait_dscnt 0x2
	v_add_f32_e32 v153, v153, v185
	s_wait_dscnt 0x1
	v_add_f32_e32 v154, v154, v186
	s_wait_dscnt 0x0
	v_add_f32_e32 v155, v155, v187
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v172, v3, v124
	ds_bpermute_b32 v173, v3, v125
	ds_bpermute_b32 v174, v3, v126
	ds_bpermute_b32 v175, v3, v127
	ds_bpermute_b32 v176, v3, v128
	ds_bpermute_b32 v177, v3, v129
	ds_bpermute_b32 v178, v3, v130
	ds_bpermute_b32 v179, v3, v131
	ds_bpermute_b32 v180, v3, v148
	ds_bpermute_b32 v181, v3, v149
	ds_bpermute_b32 v182, v3, v150
	ds_bpermute_b32 v183, v3, v151
	ds_bpermute_b32 v184, v3, v152
	ds_bpermute_b32 v185, v3, v153
	ds_bpermute_b32 v186, v3, v154
	ds_bpermute_b32 v187, v3, v155
	s_wait_dscnt 0xf
	v_add_f32_e32 v124, v124, v172
	s_wait_dscnt 0xe
	v_add_f32_e32 v125, v125, v173
	s_wait_dscnt 0xd
	v_add_f32_e32 v126, v126, v174
	s_wait_dscnt 0xc
	v_add_f32_e32 v127, v127, v175
	s_wait_dscnt 0xb
	v_add_f32_e32 v128, v128, v176
	s_wait_dscnt 0xa
	v_add_f32_e32 v129, v129, v177
	s_wait_dscnt 0x9
	v_add_f32_e32 v130, v130, v178
	s_wait_dscnt 0x8
	v_add_f32_e32 v131, v131, v179
	s_wait_dscnt 0x7
	v_add_f32_e32 v148, v148, v180
	s_wait_dscnt 0x6
	v_add_f32_e32 v149, v149, v181
	s_wait_dscnt 0x5
	v_add_f32_e32 v150, v150, v182
	s_wait_dscnt 0x4
	v_add_f32_e32 v151, v151, v183
	s_wait_dscnt 0x3
	v_add_f32_e32 v152, v152, v184
	s_wait_dscnt 0x2
	v_add_f32_e32 v153, v153, v185
	s_wait_dscnt 0x1
	v_add_f32_e32 v154, v154, v186
	s_wait_dscnt 0x0
	v_add_f32_e32 v155, v155, v187
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v172, v3, v124
	ds_bpermute_b32 v173, v3, v125
	ds_bpermute_b32 v174, v3, v126
	ds_bpermute_b32 v175, v3, v127
	ds_bpermute_b32 v176, v3, v128
	ds_bpermute_b32 v177, v3, v129
	ds_bpermute_b32 v178, v3, v130
	ds_bpermute_b32 v179, v3, v131
	ds_bpermute_b32 v180, v3, v148
	ds_bpermute_b32 v181, v3, v149
	ds_bpermute_b32 v182, v3, v150
	ds_bpermute_b32 v183, v3, v151
	ds_bpermute_b32 v184, v3, v152
	ds_bpermute_b32 v185, v3, v153
	ds_bpermute_b32 v186, v3, v154
	ds_bpermute_b32 v187, v3, v155
	s_wait_dscnt 0xf
	v_add_f32_e32 v124, v124, v172
	s_wait_dscnt 0xe
	v_add_f32_e32 v125, v125, v173
	s_wait_dscnt 0xd
	v_add_f32_e32 v126, v126, v174
	s_wait_dscnt 0xc
	v_add_f32_e32 v127, v127, v175
	s_wait_dscnt 0xb
	v_add_f32_e32 v128, v128, v176
	s_wait_dscnt 0xa
	v_add_f32_e32 v129, v129, v177
	s_wait_dscnt 0x9
	v_add_f32_e32 v130, v130, v178
	s_wait_dscnt 0x8
	v_add_f32_e32 v131, v131, v179
	s_wait_dscnt 0x7
	v_add_f32_e32 v148, v148, v180
	s_wait_dscnt 0x6
	v_add_f32_e32 v149, v149, v181
	s_wait_dscnt 0x5
	v_add_f32_e32 v150, v150, v182
	s_wait_dscnt 0x4
	v_add_f32_e32 v151, v151, v183
	s_wait_dscnt 0x3
	v_add_f32_e32 v152, v152, v184
	s_wait_dscnt 0x2
	v_add_f32_e32 v153, v153, v185
	s_wait_dscnt 0x1
	v_add_f32_e32 v154, v154, v186
	s_wait_dscnt 0x0
	v_add_f32_e32 v155, v155, v187
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v60, s46
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v61, s47
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b2_p0_single
	global_load_b32 v44, v60, s[8:9] offset:0
	global_load_b32 v45, v60, s[8:9] offset:4
	global_load_b32 v52, v61, s[8:9] offset:0
	global_load_b32 v53, v61, s[8:9] offset:4
	s_wait_loadcnt 0x3
	v_add_f32_e32 v44, v124, v44
	global_store_b32 v60, v44, s[8:9] offset:0
	s_wait_loadcnt 0x2
	v_add_f32_e32 v45, v125, v45
	global_store_b32 v60, v45, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v52, v148, v52
	global_store_b32 v61, v52, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v53, v149, v53
	global_store_b32 v61, v53, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b2_p0_next
	.Lrx_b2_p0_single:
	global_load_b32 v44, v60, s[8:9] offset:0
	global_load_b32 v52, v61, s[8:9] offset:0
	s_wait_loadcnt 0x1
	v_add_f32_e32 v44, v44, v124
	global_store_b32 v60, v44, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v52, v52, v148
	global_store_b32 v61, v52, s[8:9] offset:0
	s_wait_storecnt 0x0
	s_branch .Lrx_b2_stored
	.Lrx_b2_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b2_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b2_p1_single
	global_load_b32 v46, v60, s[8:9] offset:8
	global_load_b32 v47, v60, s[8:9] offset:12
	global_load_b32 v54, v61, s[8:9] offset:8
	global_load_b32 v55, v61, s[8:9] offset:12
	s_wait_loadcnt 0x3
	v_add_f32_e32 v46, v126, v46
	global_store_b32 v60, v46, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v47, v127, v47
	global_store_b32 v60, v47, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v54, v150, v54
	global_store_b32 v61, v54, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v55, v151, v55
	global_store_b32 v61, v55, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b2_p1_next
	.Lrx_b2_p1_single:
	global_load_b32 v46, v60, s[8:9] offset:8
	global_load_b32 v54, v61, s[8:9] offset:8
	s_wait_loadcnt 0x1
	v_add_f32_e32 v46, v46, v126
	global_store_b32 v60, v46, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v54, v54, v150
	global_store_b32 v61, v54, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b2_stored
	.Lrx_b2_p1_next:
	s_add_co_i32 s13, s12, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b2_stored
	s_add_co_i32 s13, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b2_p2_single
	global_load_b32 v48, v60, s[8:9] offset:16
	global_load_b32 v49, v60, s[8:9] offset:20
	global_load_b32 v56, v61, s[8:9] offset:16
	global_load_b32 v57, v61, s[8:9] offset:20
	s_wait_loadcnt 0x3
	v_add_f32_e32 v48, v128, v48
	global_store_b32 v60, v48, s[8:9] offset:16
	s_wait_loadcnt 0x2
	v_add_f32_e32 v49, v129, v49
	global_store_b32 v60, v49, s[8:9] offset:20
	s_wait_loadcnt 0x1
	v_add_f32_e32 v56, v152, v56
	global_store_b32 v61, v56, s[8:9] offset:16
	s_wait_loadcnt 0x0
	v_add_f32_e32 v57, v153, v57
	global_store_b32 v61, v57, s[8:9] offset:20
	s_wait_storecnt 0x0
	s_branch .Lrx_b2_p2_next
	.Lrx_b2_p2_single:
	global_load_b32 v48, v60, s[8:9] offset:16
	global_load_b32 v56, v61, s[8:9] offset:16
	s_wait_loadcnt 0x1
	v_add_f32_e32 v48, v48, v128
	global_store_b32 v60, v48, s[8:9] offset:16
	s_wait_loadcnt 0x0
	v_add_f32_e32 v56, v56, v152
	global_store_b32 v61, v56, s[8:9] offset:16
	s_wait_storecnt 0x0
	s_branch .Lrx_b2_stored
	.Lrx_b2_p2_next:
	s_add_co_i32 s13, s12, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b2_stored
	s_add_co_i32 s13, s12, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b2_p3_single
	global_load_b32 v50, v60, s[8:9] offset:24
	global_load_b32 v51, v60, s[8:9] offset:28
	global_load_b32 v58, v61, s[8:9] offset:24
	global_load_b32 v59, v61, s[8:9] offset:28
	s_wait_loadcnt 0x3
	v_add_f32_e32 v50, v130, v50
	global_store_b32 v60, v50, s[8:9] offset:24
	s_wait_loadcnt 0x2
	v_add_f32_e32 v51, v131, v51
	global_store_b32 v60, v51, s[8:9] offset:28
	s_wait_loadcnt 0x1
	v_add_f32_e32 v58, v154, v58
	global_store_b32 v61, v58, s[8:9] offset:24
	s_wait_loadcnt 0x0
	v_add_f32_e32 v59, v155, v59
	global_store_b32 v61, v59, s[8:9] offset:28
	s_wait_storecnt 0x0
	s_branch .Lrx_b2_stored
	.Lrx_b2_p3_single:
	global_load_b32 v50, v60, s[8:9] offset:24
	global_load_b32 v58, v61, s[8:9] offset:24
	s_wait_loadcnt 0x1
	v_add_f32_e32 v50, v50, v130
	global_store_b32 v60, v50, s[8:9] offset:24
	s_wait_loadcnt 0x0
	v_add_f32_e32 v58, v58, v154
	global_store_b32 v61, v58, s[8:9] offset:24
	s_wait_storecnt 0x0
	s_branch .Lrx_b2_stored
	.Lrx_b2_stored:
	s_branch .Lrx_end
	.Lrx_b3:
	s_mul_i32 s12, ttmp9, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v12, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v13, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v14, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v15, s13
	v_add_nc_u32_e32 v7, s13, v1
	s_add_co_i32 s13, s12, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v16, s13
	v_add_nc_u32_e32 v8, s13, v1
	s_add_co_i32 s13, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v17, s13
	v_add_nc_u32_e32 v9, s13, v1
	s_add_co_i32 s13, s12, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v18, s13
	v_add_nc_u32_e32 v10, s13, v1
	s_add_co_i32 s13, s12, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v19, s13
	v_add_nc_u32_e32 v11, s13, v1
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s0_done
	s_mov_b32 s15, 0x0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[196:199], v2, s[28:29]
	global_load_b128 v[200:203], v2, s[28:29] offset:16
	global_load_b128 v[204:207], v2, s[30:31]
	global_load_b128 v[208:211], v2, s[30:31] offset:16
	global_load_b128 v[212:215], v2, s[32:33]
	global_load_b128 v[216:219], v2, s[32:33] offset:16
	.Lrx_b3_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x6
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v44, v196 :: v_dual_mul_f32 v53, v55, v197
	v_dual_mul_f32 v62, v64, v196 :: v_dual_mul_f32 v63, v75, v197
	v_dual_mul_f32 v72, v84, v196 :: v_dual_mul_f32 v73, v95, v197
	v_dual_mul_f32 v82, v104, v196 :: v_dual_mul_f32 v83, v115, v197
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v92, v44, v204 :: v_dual_mul_f32 v93, v55, v205
	v_dual_mul_f32 v102, v64, v204 :: v_dual_mul_f32 v103, v75, v205
	v_dual_mul_f32 v112, v84, v204 :: v_dual_mul_f32 v113, v95, v205
	v_dual_mul_f32 v122, v104, v204 :: v_dual_mul_f32 v123, v115, v205
	v_dual_fmac_f32 v52, v45, v197 :: v_dual_fmac_f32 v53, v54, v196
	v_dual_fmac_f32 v62, v65, v197 :: v_dual_fmac_f32 v63, v74, v196
	v_dual_fmac_f32 v72, v85, v197 :: v_dual_fmac_f32 v73, v94, v196
	v_dual_fmac_f32 v82, v105, v197 :: v_dual_fmac_f32 v83, v114, v196
	v_dual_fmac_f32 v92, v45, v205 :: v_dual_fmac_f32 v93, v54, v204
	v_dual_fmac_f32 v102, v65, v205 :: v_dual_fmac_f32 v103, v74, v204
	v_dual_fmac_f32 v112, v85, v205 :: v_dual_fmac_f32 v113, v94, v204
	v_dual_fmac_f32 v122, v105, v205 :: v_dual_fmac_f32 v123, v114, v204
	v_dual_fmac_f32 v52, v46, v198 :: v_dual_fmac_f32 v53, v56, v198
	v_dual_fmac_f32 v62, v66, v198 :: v_dual_fmac_f32 v63, v76, v198
	v_dual_fmac_f32 v72, v86, v198 :: v_dual_fmac_f32 v73, v96, v198
	v_dual_fmac_f32 v82, v106, v198 :: v_dual_fmac_f32 v83, v116, v198
	v_dual_fmac_f32 v92, v46, v206 :: v_dual_fmac_f32 v93, v56, v206
	v_dual_fmac_f32 v102, v66, v206 :: v_dual_fmac_f32 v103, v76, v206
	v_dual_fmac_f32 v112, v86, v206 :: v_dual_fmac_f32 v113, v96, v206
	v_dual_fmac_f32 v122, v106, v206 :: v_dual_fmac_f32 v123, v116, v206
	v_dual_fmac_f32 v52, v47, v199 :: v_dual_fmac_f32 v53, v57, v199
	v_dual_fmac_f32 v62, v67, v199 :: v_dual_fmac_f32 v63, v77, v199
	v_dual_fmac_f32 v72, v87, v199 :: v_dual_fmac_f32 v73, v97, v199
	v_dual_fmac_f32 v82, v107, v199 :: v_dual_fmac_f32 v83, v117, v199
	v_dual_fmac_f32 v92, v47, v207 :: v_dual_fmac_f32 v93, v57, v207
	v_dual_fmac_f32 v102, v67, v207 :: v_dual_fmac_f32 v103, v77, v207
	v_dual_fmac_f32 v112, v87, v207 :: v_dual_fmac_f32 v113, v97, v207
	v_dual_fmac_f32 v122, v107, v207 :: v_dual_fmac_f32 v123, v117, v207
	v_dual_fmac_f32 v52, v48, v200 :: v_dual_fmac_f32 v53, v58, v200
	v_dual_fmac_f32 v62, v68, v200 :: v_dual_fmac_f32 v63, v78, v200
	v_dual_fmac_f32 v72, v88, v200 :: v_dual_fmac_f32 v73, v98, v200
	v_dual_fmac_f32 v82, v108, v200 :: v_dual_fmac_f32 v83, v118, v200
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v92, v48, v208 :: v_dual_fmac_f32 v93, v58, v208
	v_dual_fmac_f32 v102, v68, v208 :: v_dual_fmac_f32 v103, v78, v208
	v_dual_fmac_f32 v112, v88, v208 :: v_dual_fmac_f32 v113, v98, v208
	v_dual_fmac_f32 v122, v108, v208 :: v_dual_fmac_f32 v123, v118, v208
	v_dual_fmac_f32 v52, v49, v201 :: v_dual_fmac_f32 v53, v59, v201
	v_dual_fmac_f32 v62, v69, v201 :: v_dual_fmac_f32 v63, v79, v201
	v_dual_fmac_f32 v72, v89, v201 :: v_dual_fmac_f32 v73, v99, v201
	v_dual_fmac_f32 v82, v109, v201 :: v_dual_fmac_f32 v83, v119, v201
	v_dual_fmac_f32 v92, v49, v209 :: v_dual_fmac_f32 v93, v59, v209
	v_dual_fmac_f32 v102, v69, v209 :: v_dual_fmac_f32 v103, v79, v209
	v_dual_fmac_f32 v112, v89, v209 :: v_dual_fmac_f32 v113, v99, v209
	v_dual_fmac_f32 v122, v109, v209 :: v_dual_fmac_f32 v123, v119, v209
	v_dual_fmac_f32 v52, v50, v202 :: v_dual_fmac_f32 v53, v60, v202
	v_dual_fmac_f32 v62, v70, v202 :: v_dual_fmac_f32 v63, v80, v202
	v_dual_fmac_f32 v72, v90, v202 :: v_dual_fmac_f32 v73, v100, v202
	v_dual_fmac_f32 v82, v110, v202 :: v_dual_fmac_f32 v83, v120, v202
	v_dual_fmac_f32 v92, v50, v210 :: v_dual_fmac_f32 v93, v60, v210
	v_dual_fmac_f32 v102, v70, v210 :: v_dual_fmac_f32 v103, v80, v210
	v_dual_fmac_f32 v112, v90, v210 :: v_dual_fmac_f32 v113, v100, v210
	v_dual_fmac_f32 v122, v110, v210 :: v_dual_fmac_f32 v123, v120, v210
	v_dual_fmac_f32 v52, v51, v203 :: v_dual_fmac_f32 v53, v61, v203
	v_dual_fmac_f32 v62, v71, v203 :: v_dual_fmac_f32 v63, v81, v203
	v_dual_fmac_f32 v72, v91, v203 :: v_dual_fmac_f32 v73, v101, v203
	v_dual_fmac_f32 v82, v111, v203 :: v_dual_fmac_f32 v83, v121, v203
	v_dual_fmac_f32 v92, v51, v211 :: v_dual_fmac_f32 v93, v61, v211
	v_dual_fmac_f32 v102, v71, v211 :: v_dual_fmac_f32 v103, v81, v211
	v_dual_fmac_f32 v112, v91, v211 :: v_dual_fmac_f32 v113, v101, v211
	v_dual_fmac_f32 v122, v111, v211 :: v_dual_fmac_f32 v123, v121, v211
	global_load_b128 v[196:199], v3, s[28:29]
	global_load_b128 v[200:203], v3, s[28:29] offset:16
	global_load_b128 v[204:207], v3, s[30:31]
	global_load_b128 v[208:211], v3, s[30:31] offset:16
	v_dual_add_f32 v124, v52, v124 :: v_dual_add_f32 v125, v125, v53
	v_dual_add_f32 v126, v62, v126 :: v_dual_add_f32 v127, v127, v63
	v_dual_add_f32 v128, v72, v128 :: v_dual_add_f32 v129, v129, v73
	v_dual_add_f32 v130, v82, v130 :: v_dual_add_f32 v131, v131, v83
	v_dual_add_f32 v148, v92, v148 :: v_dual_add_f32 v149, v149, v93
	v_dual_add_f32 v150, v102, v150 :: v_dual_add_f32 v151, v151, v103
	v_dual_add_f32 v152, v112, v152 :: v_dual_add_f32 v153, v153, v113
	v_dual_add_f32 v154, v122, v154 :: v_dual_add_f32 v155, v155, v123
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v44, v212 :: v_dual_mul_f32 v53, v55, v213
	v_dual_mul_f32 v62, v64, v212 :: v_dual_mul_f32 v63, v75, v213
	v_dual_mul_f32 v72, v84, v212 :: v_dual_mul_f32 v73, v95, v213
	v_dual_mul_f32 v82, v104, v212 :: v_dual_mul_f32 v83, v115, v213
	v_dual_fmac_f32 v52, v45, v213 :: v_dual_fmac_f32 v53, v54, v212
	v_dual_fmac_f32 v62, v65, v213 :: v_dual_fmac_f32 v63, v74, v212
	v_dual_fmac_f32 v72, v85, v213 :: v_dual_fmac_f32 v73, v94, v212
	v_dual_fmac_f32 v82, v105, v213 :: v_dual_fmac_f32 v83, v114, v212
	v_dual_fmac_f32 v52, v46, v214 :: v_dual_fmac_f32 v53, v56, v214
	v_dual_fmac_f32 v62, v66, v214 :: v_dual_fmac_f32 v63, v76, v214
	v_dual_fmac_f32 v72, v86, v214 :: v_dual_fmac_f32 v73, v96, v214
	v_dual_fmac_f32 v82, v106, v214 :: v_dual_fmac_f32 v83, v116, v214
	v_dual_fmac_f32 v52, v47, v215 :: v_dual_fmac_f32 v53, v57, v215
	v_dual_fmac_f32 v62, v67, v215 :: v_dual_fmac_f32 v63, v77, v215
	v_dual_fmac_f32 v72, v87, v215 :: v_dual_fmac_f32 v73, v97, v215
	v_dual_fmac_f32 v82, v107, v215 :: v_dual_fmac_f32 v83, v117, v215
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v48, v216 :: v_dual_fmac_f32 v53, v58, v216
	v_dual_fmac_f32 v62, v68, v216 :: v_dual_fmac_f32 v63, v78, v216
	v_dual_fmac_f32 v72, v88, v216 :: v_dual_fmac_f32 v73, v98, v216
	v_dual_fmac_f32 v82, v108, v216 :: v_dual_fmac_f32 v83, v118, v216
	v_dual_fmac_f32 v52, v49, v217 :: v_dual_fmac_f32 v53, v59, v217
	v_dual_fmac_f32 v62, v69, v217 :: v_dual_fmac_f32 v63, v79, v217
	v_dual_fmac_f32 v72, v89, v217 :: v_dual_fmac_f32 v73, v99, v217
	v_dual_fmac_f32 v82, v109, v217 :: v_dual_fmac_f32 v83, v119, v217
	v_dual_fmac_f32 v52, v50, v218 :: v_dual_fmac_f32 v53, v60, v218
	v_dual_fmac_f32 v62, v70, v218 :: v_dual_fmac_f32 v63, v80, v218
	v_dual_fmac_f32 v72, v90, v218 :: v_dual_fmac_f32 v73, v100, v218
	v_dual_fmac_f32 v82, v110, v218 :: v_dual_fmac_f32 v83, v120, v218
	v_dual_fmac_f32 v52, v51, v219 :: v_dual_fmac_f32 v53, v61, v219
	v_dual_fmac_f32 v62, v71, v219 :: v_dual_fmac_f32 v63, v81, v219
	v_dual_fmac_f32 v72, v91, v219 :: v_dual_fmac_f32 v73, v101, v219
	v_dual_fmac_f32 v82, v111, v219 :: v_dual_fmac_f32 v83, v121, v219
	global_load_b128 v[212:215], v3, s[32:33]
	global_load_b128 v[216:219], v3, s[32:33] offset:16
	v_dual_add_f32 v172, v52, v172 :: v_dual_add_f32 v173, v173, v53
	v_dual_add_f32 v174, v62, v174 :: v_dual_add_f32 v175, v175, v63
	v_dual_add_f32 v176, v72, v176 :: v_dual_add_f32 v177, v177, v73
	v_dual_add_f32 v178, v82, v178 :: v_dual_add_f32 v179, v179, v83
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s0
	s_wait_loadcnt 0x0
	.Lrx_b3_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[196:199], v2, s[28:29]
	global_load_b128 v[200:203], v2, s[28:29] offset:16
	global_load_b128 v[204:207], v2, s[30:31]
	global_load_b128 v[208:211], v2, s[30:31] offset:16
	global_load_b128 v[212:215], v2, s[32:33]
	global_load_b128 v[216:219], v2, s[32:33] offset:16
	.Lrx_b3_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x6
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v44, v196 :: v_dual_mul_f32 v53, v55, v197
	v_dual_mul_f32 v62, v64, v196 :: v_dual_mul_f32 v63, v75, v197
	v_dual_mul_f32 v72, v84, v196 :: v_dual_mul_f32 v73, v95, v197
	v_dual_mul_f32 v82, v104, v196 :: v_dual_mul_f32 v83, v115, v197
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v92, v44, v204 :: v_dual_mul_f32 v93, v55, v205
	v_dual_mul_f32 v102, v64, v204 :: v_dual_mul_f32 v103, v75, v205
	v_dual_mul_f32 v112, v84, v204 :: v_dual_mul_f32 v113, v95, v205
	v_dual_mul_f32 v122, v104, v204 :: v_dual_mul_f32 v123, v115, v205
	v_dual_fmac_f32 v52, v45, v197 :: v_dual_fmac_f32 v53, v54, v196
	v_dual_fmac_f32 v62, v65, v197 :: v_dual_fmac_f32 v63, v74, v196
	v_dual_fmac_f32 v72, v85, v197 :: v_dual_fmac_f32 v73, v94, v196
	v_dual_fmac_f32 v82, v105, v197 :: v_dual_fmac_f32 v83, v114, v196
	v_dual_fmac_f32 v92, v45, v205 :: v_dual_fmac_f32 v93, v54, v204
	v_dual_fmac_f32 v102, v65, v205 :: v_dual_fmac_f32 v103, v74, v204
	v_dual_fmac_f32 v112, v85, v205 :: v_dual_fmac_f32 v113, v94, v204
	v_dual_fmac_f32 v122, v105, v205 :: v_dual_fmac_f32 v123, v114, v204
	v_dual_fmac_f32 v52, v46, v198 :: v_dual_fmac_f32 v53, v56, v198
	v_dual_fmac_f32 v62, v66, v198 :: v_dual_fmac_f32 v63, v76, v198
	v_dual_fmac_f32 v72, v86, v198 :: v_dual_fmac_f32 v73, v96, v198
	v_dual_fmac_f32 v82, v106, v198 :: v_dual_fmac_f32 v83, v116, v198
	v_dual_fmac_f32 v92, v46, v206 :: v_dual_fmac_f32 v93, v56, v206
	v_dual_fmac_f32 v102, v66, v206 :: v_dual_fmac_f32 v103, v76, v206
	v_dual_fmac_f32 v112, v86, v206 :: v_dual_fmac_f32 v113, v96, v206
	v_dual_fmac_f32 v122, v106, v206 :: v_dual_fmac_f32 v123, v116, v206
	v_dual_fmac_f32 v52, v47, v199 :: v_dual_fmac_f32 v53, v57, v199
	v_dual_fmac_f32 v62, v67, v199 :: v_dual_fmac_f32 v63, v77, v199
	v_dual_fmac_f32 v72, v87, v199 :: v_dual_fmac_f32 v73, v97, v199
	v_dual_fmac_f32 v82, v107, v199 :: v_dual_fmac_f32 v83, v117, v199
	v_dual_fmac_f32 v92, v47, v207 :: v_dual_fmac_f32 v93, v57, v207
	v_dual_fmac_f32 v102, v67, v207 :: v_dual_fmac_f32 v103, v77, v207
	v_dual_fmac_f32 v112, v87, v207 :: v_dual_fmac_f32 v113, v97, v207
	v_dual_fmac_f32 v122, v107, v207 :: v_dual_fmac_f32 v123, v117, v207
	v_dual_fmac_f32 v52, v48, v200 :: v_dual_fmac_f32 v53, v58, v200
	v_dual_fmac_f32 v62, v68, v200 :: v_dual_fmac_f32 v63, v78, v200
	v_dual_fmac_f32 v72, v88, v200 :: v_dual_fmac_f32 v73, v98, v200
	v_dual_fmac_f32 v82, v108, v200 :: v_dual_fmac_f32 v83, v118, v200
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v92, v48, v208 :: v_dual_fmac_f32 v93, v58, v208
	v_dual_fmac_f32 v102, v68, v208 :: v_dual_fmac_f32 v103, v78, v208
	v_dual_fmac_f32 v112, v88, v208 :: v_dual_fmac_f32 v113, v98, v208
	v_dual_fmac_f32 v122, v108, v208 :: v_dual_fmac_f32 v123, v118, v208
	v_dual_fmac_f32 v52, v49, v201 :: v_dual_fmac_f32 v53, v59, v201
	v_dual_fmac_f32 v62, v69, v201 :: v_dual_fmac_f32 v63, v79, v201
	v_dual_fmac_f32 v72, v89, v201 :: v_dual_fmac_f32 v73, v99, v201
	v_dual_fmac_f32 v82, v109, v201 :: v_dual_fmac_f32 v83, v119, v201
	v_dual_fmac_f32 v92, v49, v209 :: v_dual_fmac_f32 v93, v59, v209
	v_dual_fmac_f32 v102, v69, v209 :: v_dual_fmac_f32 v103, v79, v209
	v_dual_fmac_f32 v112, v89, v209 :: v_dual_fmac_f32 v113, v99, v209
	v_dual_fmac_f32 v122, v109, v209 :: v_dual_fmac_f32 v123, v119, v209
	v_dual_fmac_f32 v52, v50, v202 :: v_dual_fmac_f32 v53, v60, v202
	v_dual_fmac_f32 v62, v70, v202 :: v_dual_fmac_f32 v63, v80, v202
	v_dual_fmac_f32 v72, v90, v202 :: v_dual_fmac_f32 v73, v100, v202
	v_dual_fmac_f32 v82, v110, v202 :: v_dual_fmac_f32 v83, v120, v202
	v_dual_fmac_f32 v92, v50, v210 :: v_dual_fmac_f32 v93, v60, v210
	v_dual_fmac_f32 v102, v70, v210 :: v_dual_fmac_f32 v103, v80, v210
	v_dual_fmac_f32 v112, v90, v210 :: v_dual_fmac_f32 v113, v100, v210
	v_dual_fmac_f32 v122, v110, v210 :: v_dual_fmac_f32 v123, v120, v210
	v_dual_fmac_f32 v52, v51, v203 :: v_dual_fmac_f32 v53, v61, v203
	v_dual_fmac_f32 v62, v71, v203 :: v_dual_fmac_f32 v63, v81, v203
	v_dual_fmac_f32 v72, v91, v203 :: v_dual_fmac_f32 v73, v101, v203
	v_dual_fmac_f32 v82, v111, v203 :: v_dual_fmac_f32 v83, v121, v203
	v_dual_fmac_f32 v92, v51, v211 :: v_dual_fmac_f32 v93, v61, v211
	v_dual_fmac_f32 v102, v71, v211 :: v_dual_fmac_f32 v103, v81, v211
	v_dual_fmac_f32 v112, v91, v211 :: v_dual_fmac_f32 v113, v101, v211
	v_dual_fmac_f32 v122, v111, v211 :: v_dual_fmac_f32 v123, v121, v211
	global_load_b128 v[196:199], v3, s[28:29]
	global_load_b128 v[200:203], v3, s[28:29] offset:16
	global_load_b128 v[204:207], v3, s[30:31]
	global_load_b128 v[208:211], v3, s[30:31] offset:16
	v_dual_add_f32 v132, v52, v132 :: v_dual_add_f32 v133, v133, v53
	v_dual_add_f32 v134, v62, v134 :: v_dual_add_f32 v135, v135, v63
	v_dual_add_f32 v136, v72, v136 :: v_dual_add_f32 v137, v137, v73
	v_dual_add_f32 v138, v82, v138 :: v_dual_add_f32 v139, v139, v83
	v_dual_add_f32 v156, v92, v156 :: v_dual_add_f32 v157, v157, v93
	v_dual_add_f32 v158, v102, v158 :: v_dual_add_f32 v159, v159, v103
	v_dual_add_f32 v160, v112, v160 :: v_dual_add_f32 v161, v161, v113
	v_dual_add_f32 v162, v122, v162 :: v_dual_add_f32 v163, v163, v123
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v44, v212 :: v_dual_mul_f32 v53, v55, v213
	v_dual_mul_f32 v62, v64, v212 :: v_dual_mul_f32 v63, v75, v213
	v_dual_mul_f32 v72, v84, v212 :: v_dual_mul_f32 v73, v95, v213
	v_dual_mul_f32 v82, v104, v212 :: v_dual_mul_f32 v83, v115, v213
	v_dual_fmac_f32 v52, v45, v213 :: v_dual_fmac_f32 v53, v54, v212
	v_dual_fmac_f32 v62, v65, v213 :: v_dual_fmac_f32 v63, v74, v212
	v_dual_fmac_f32 v72, v85, v213 :: v_dual_fmac_f32 v73, v94, v212
	v_dual_fmac_f32 v82, v105, v213 :: v_dual_fmac_f32 v83, v114, v212
	v_dual_fmac_f32 v52, v46, v214 :: v_dual_fmac_f32 v53, v56, v214
	v_dual_fmac_f32 v62, v66, v214 :: v_dual_fmac_f32 v63, v76, v214
	v_dual_fmac_f32 v72, v86, v214 :: v_dual_fmac_f32 v73, v96, v214
	v_dual_fmac_f32 v82, v106, v214 :: v_dual_fmac_f32 v83, v116, v214
	v_dual_fmac_f32 v52, v47, v215 :: v_dual_fmac_f32 v53, v57, v215
	v_dual_fmac_f32 v62, v67, v215 :: v_dual_fmac_f32 v63, v77, v215
	v_dual_fmac_f32 v72, v87, v215 :: v_dual_fmac_f32 v73, v97, v215
	v_dual_fmac_f32 v82, v107, v215 :: v_dual_fmac_f32 v83, v117, v215
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v48, v216 :: v_dual_fmac_f32 v53, v58, v216
	v_dual_fmac_f32 v62, v68, v216 :: v_dual_fmac_f32 v63, v78, v216
	v_dual_fmac_f32 v72, v88, v216 :: v_dual_fmac_f32 v73, v98, v216
	v_dual_fmac_f32 v82, v108, v216 :: v_dual_fmac_f32 v83, v118, v216
	v_dual_fmac_f32 v52, v49, v217 :: v_dual_fmac_f32 v53, v59, v217
	v_dual_fmac_f32 v62, v69, v217 :: v_dual_fmac_f32 v63, v79, v217
	v_dual_fmac_f32 v72, v89, v217 :: v_dual_fmac_f32 v73, v99, v217
	v_dual_fmac_f32 v82, v109, v217 :: v_dual_fmac_f32 v83, v119, v217
	v_dual_fmac_f32 v52, v50, v218 :: v_dual_fmac_f32 v53, v60, v218
	v_dual_fmac_f32 v62, v70, v218 :: v_dual_fmac_f32 v63, v80, v218
	v_dual_fmac_f32 v72, v90, v218 :: v_dual_fmac_f32 v73, v100, v218
	v_dual_fmac_f32 v82, v110, v218 :: v_dual_fmac_f32 v83, v120, v218
	v_dual_fmac_f32 v52, v51, v219 :: v_dual_fmac_f32 v53, v61, v219
	v_dual_fmac_f32 v62, v71, v219 :: v_dual_fmac_f32 v63, v81, v219
	v_dual_fmac_f32 v72, v91, v219 :: v_dual_fmac_f32 v73, v101, v219
	v_dual_fmac_f32 v82, v111, v219 :: v_dual_fmac_f32 v83, v121, v219
	global_load_b128 v[212:215], v3, s[32:33]
	global_load_b128 v[216:219], v3, s[32:33] offset:16
	v_dual_add_f32 v180, v52, v180 :: v_dual_add_f32 v181, v181, v53
	v_dual_add_f32 v182, v62, v182 :: v_dual_add_f32 v183, v183, v63
	v_dual_add_f32 v184, v72, v184 :: v_dual_add_f32 v185, v185, v73
	v_dual_add_f32 v186, v82, v186 :: v_dual_add_f32 v187, v187, v83
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s1
	s_wait_loadcnt 0x0
	.Lrx_b3_s1_done:
	v_dual_add_f32 v124, v124, v132 :: v_dual_add_f32 v125, v125, v133
	v_dual_add_f32 v126, v126, v134 :: v_dual_add_f32 v127, v127, v135
	v_dual_add_f32 v128, v128, v136 :: v_dual_add_f32 v129, v129, v137
	v_dual_add_f32 v130, v130, v138 :: v_dual_add_f32 v131, v131, v139
	v_dual_add_f32 v148, v148, v156 :: v_dual_add_f32 v149, v149, v157
	v_dual_add_f32 v150, v150, v158 :: v_dual_add_f32 v151, v151, v159
	v_dual_add_f32 v152, v152, v160 :: v_dual_add_f32 v153, v153, v161
	v_dual_add_f32 v154, v154, v162 :: v_dual_add_f32 v155, v155, v163
	v_dual_add_f32 v172, v172, v180 :: v_dual_add_f32 v173, v173, v181
	v_dual_add_f32 v174, v174, v182 :: v_dual_add_f32 v175, v175, v183
	v_dual_add_f32 v176, v176, v184 :: v_dual_add_f32 v177, v177, v185
	v_dual_add_f32 v178, v178, v186 :: v_dual_add_f32 v179, v179, v187
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[196:199], v2, s[28:29]
	global_load_b128 v[200:203], v2, s[28:29] offset:16
	global_load_b128 v[204:207], v2, s[30:31]
	global_load_b128 v[208:211], v2, s[30:31] offset:16
	global_load_b128 v[212:215], v2, s[32:33]
	global_load_b128 v[216:219], v2, s[32:33] offset:16
	.Lrx_b3_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x6
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v44, v196 :: v_dual_mul_f32 v53, v55, v197
	v_dual_mul_f32 v62, v64, v196 :: v_dual_mul_f32 v63, v75, v197
	v_dual_mul_f32 v72, v84, v196 :: v_dual_mul_f32 v73, v95, v197
	v_dual_mul_f32 v82, v104, v196 :: v_dual_mul_f32 v83, v115, v197
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v92, v44, v204 :: v_dual_mul_f32 v93, v55, v205
	v_dual_mul_f32 v102, v64, v204 :: v_dual_mul_f32 v103, v75, v205
	v_dual_mul_f32 v112, v84, v204 :: v_dual_mul_f32 v113, v95, v205
	v_dual_mul_f32 v122, v104, v204 :: v_dual_mul_f32 v123, v115, v205
	v_dual_fmac_f32 v52, v45, v197 :: v_dual_fmac_f32 v53, v54, v196
	v_dual_fmac_f32 v62, v65, v197 :: v_dual_fmac_f32 v63, v74, v196
	v_dual_fmac_f32 v72, v85, v197 :: v_dual_fmac_f32 v73, v94, v196
	v_dual_fmac_f32 v82, v105, v197 :: v_dual_fmac_f32 v83, v114, v196
	v_dual_fmac_f32 v92, v45, v205 :: v_dual_fmac_f32 v93, v54, v204
	v_dual_fmac_f32 v102, v65, v205 :: v_dual_fmac_f32 v103, v74, v204
	v_dual_fmac_f32 v112, v85, v205 :: v_dual_fmac_f32 v113, v94, v204
	v_dual_fmac_f32 v122, v105, v205 :: v_dual_fmac_f32 v123, v114, v204
	v_dual_fmac_f32 v52, v46, v198 :: v_dual_fmac_f32 v53, v56, v198
	v_dual_fmac_f32 v62, v66, v198 :: v_dual_fmac_f32 v63, v76, v198
	v_dual_fmac_f32 v72, v86, v198 :: v_dual_fmac_f32 v73, v96, v198
	v_dual_fmac_f32 v82, v106, v198 :: v_dual_fmac_f32 v83, v116, v198
	v_dual_fmac_f32 v92, v46, v206 :: v_dual_fmac_f32 v93, v56, v206
	v_dual_fmac_f32 v102, v66, v206 :: v_dual_fmac_f32 v103, v76, v206
	v_dual_fmac_f32 v112, v86, v206 :: v_dual_fmac_f32 v113, v96, v206
	v_dual_fmac_f32 v122, v106, v206 :: v_dual_fmac_f32 v123, v116, v206
	v_dual_fmac_f32 v52, v47, v199 :: v_dual_fmac_f32 v53, v57, v199
	v_dual_fmac_f32 v62, v67, v199 :: v_dual_fmac_f32 v63, v77, v199
	v_dual_fmac_f32 v72, v87, v199 :: v_dual_fmac_f32 v73, v97, v199
	v_dual_fmac_f32 v82, v107, v199 :: v_dual_fmac_f32 v83, v117, v199
	v_dual_fmac_f32 v92, v47, v207 :: v_dual_fmac_f32 v93, v57, v207
	v_dual_fmac_f32 v102, v67, v207 :: v_dual_fmac_f32 v103, v77, v207
	v_dual_fmac_f32 v112, v87, v207 :: v_dual_fmac_f32 v113, v97, v207
	v_dual_fmac_f32 v122, v107, v207 :: v_dual_fmac_f32 v123, v117, v207
	v_dual_fmac_f32 v52, v48, v200 :: v_dual_fmac_f32 v53, v58, v200
	v_dual_fmac_f32 v62, v68, v200 :: v_dual_fmac_f32 v63, v78, v200
	v_dual_fmac_f32 v72, v88, v200 :: v_dual_fmac_f32 v73, v98, v200
	v_dual_fmac_f32 v82, v108, v200 :: v_dual_fmac_f32 v83, v118, v200
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v92, v48, v208 :: v_dual_fmac_f32 v93, v58, v208
	v_dual_fmac_f32 v102, v68, v208 :: v_dual_fmac_f32 v103, v78, v208
	v_dual_fmac_f32 v112, v88, v208 :: v_dual_fmac_f32 v113, v98, v208
	v_dual_fmac_f32 v122, v108, v208 :: v_dual_fmac_f32 v123, v118, v208
	v_dual_fmac_f32 v52, v49, v201 :: v_dual_fmac_f32 v53, v59, v201
	v_dual_fmac_f32 v62, v69, v201 :: v_dual_fmac_f32 v63, v79, v201
	v_dual_fmac_f32 v72, v89, v201 :: v_dual_fmac_f32 v73, v99, v201
	v_dual_fmac_f32 v82, v109, v201 :: v_dual_fmac_f32 v83, v119, v201
	v_dual_fmac_f32 v92, v49, v209 :: v_dual_fmac_f32 v93, v59, v209
	v_dual_fmac_f32 v102, v69, v209 :: v_dual_fmac_f32 v103, v79, v209
	v_dual_fmac_f32 v112, v89, v209 :: v_dual_fmac_f32 v113, v99, v209
	v_dual_fmac_f32 v122, v109, v209 :: v_dual_fmac_f32 v123, v119, v209
	v_dual_fmac_f32 v52, v50, v202 :: v_dual_fmac_f32 v53, v60, v202
	v_dual_fmac_f32 v62, v70, v202 :: v_dual_fmac_f32 v63, v80, v202
	v_dual_fmac_f32 v72, v90, v202 :: v_dual_fmac_f32 v73, v100, v202
	v_dual_fmac_f32 v82, v110, v202 :: v_dual_fmac_f32 v83, v120, v202
	v_dual_fmac_f32 v92, v50, v210 :: v_dual_fmac_f32 v93, v60, v210
	v_dual_fmac_f32 v102, v70, v210 :: v_dual_fmac_f32 v103, v80, v210
	v_dual_fmac_f32 v112, v90, v210 :: v_dual_fmac_f32 v113, v100, v210
	v_dual_fmac_f32 v122, v110, v210 :: v_dual_fmac_f32 v123, v120, v210
	v_dual_fmac_f32 v52, v51, v203 :: v_dual_fmac_f32 v53, v61, v203
	v_dual_fmac_f32 v62, v71, v203 :: v_dual_fmac_f32 v63, v81, v203
	v_dual_fmac_f32 v72, v91, v203 :: v_dual_fmac_f32 v73, v101, v203
	v_dual_fmac_f32 v82, v111, v203 :: v_dual_fmac_f32 v83, v121, v203
	v_dual_fmac_f32 v92, v51, v211 :: v_dual_fmac_f32 v93, v61, v211
	v_dual_fmac_f32 v102, v71, v211 :: v_dual_fmac_f32 v103, v81, v211
	v_dual_fmac_f32 v112, v91, v211 :: v_dual_fmac_f32 v113, v101, v211
	v_dual_fmac_f32 v122, v111, v211 :: v_dual_fmac_f32 v123, v121, v211
	global_load_b128 v[196:199], v3, s[28:29]
	global_load_b128 v[200:203], v3, s[28:29] offset:16
	global_load_b128 v[204:207], v3, s[30:31]
	global_load_b128 v[208:211], v3, s[30:31] offset:16
	v_dual_add_f32 v132, v52, v132 :: v_dual_add_f32 v133, v133, v53
	v_dual_add_f32 v134, v62, v134 :: v_dual_add_f32 v135, v135, v63
	v_dual_add_f32 v136, v72, v136 :: v_dual_add_f32 v137, v137, v73
	v_dual_add_f32 v138, v82, v138 :: v_dual_add_f32 v139, v139, v83
	v_dual_add_f32 v156, v92, v156 :: v_dual_add_f32 v157, v157, v93
	v_dual_add_f32 v158, v102, v158 :: v_dual_add_f32 v159, v159, v103
	v_dual_add_f32 v160, v112, v160 :: v_dual_add_f32 v161, v161, v113
	v_dual_add_f32 v162, v122, v162 :: v_dual_add_f32 v163, v163, v123
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v44, v212 :: v_dual_mul_f32 v53, v55, v213
	v_dual_mul_f32 v62, v64, v212 :: v_dual_mul_f32 v63, v75, v213
	v_dual_mul_f32 v72, v84, v212 :: v_dual_mul_f32 v73, v95, v213
	v_dual_mul_f32 v82, v104, v212 :: v_dual_mul_f32 v83, v115, v213
	v_dual_fmac_f32 v52, v45, v213 :: v_dual_fmac_f32 v53, v54, v212
	v_dual_fmac_f32 v62, v65, v213 :: v_dual_fmac_f32 v63, v74, v212
	v_dual_fmac_f32 v72, v85, v213 :: v_dual_fmac_f32 v73, v94, v212
	v_dual_fmac_f32 v82, v105, v213 :: v_dual_fmac_f32 v83, v114, v212
	v_dual_fmac_f32 v52, v46, v214 :: v_dual_fmac_f32 v53, v56, v214
	v_dual_fmac_f32 v62, v66, v214 :: v_dual_fmac_f32 v63, v76, v214
	v_dual_fmac_f32 v72, v86, v214 :: v_dual_fmac_f32 v73, v96, v214
	v_dual_fmac_f32 v82, v106, v214 :: v_dual_fmac_f32 v83, v116, v214
	v_dual_fmac_f32 v52, v47, v215 :: v_dual_fmac_f32 v53, v57, v215
	v_dual_fmac_f32 v62, v67, v215 :: v_dual_fmac_f32 v63, v77, v215
	v_dual_fmac_f32 v72, v87, v215 :: v_dual_fmac_f32 v73, v97, v215
	v_dual_fmac_f32 v82, v107, v215 :: v_dual_fmac_f32 v83, v117, v215
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v48, v216 :: v_dual_fmac_f32 v53, v58, v216
	v_dual_fmac_f32 v62, v68, v216 :: v_dual_fmac_f32 v63, v78, v216
	v_dual_fmac_f32 v72, v88, v216 :: v_dual_fmac_f32 v73, v98, v216
	v_dual_fmac_f32 v82, v108, v216 :: v_dual_fmac_f32 v83, v118, v216
	v_dual_fmac_f32 v52, v49, v217 :: v_dual_fmac_f32 v53, v59, v217
	v_dual_fmac_f32 v62, v69, v217 :: v_dual_fmac_f32 v63, v79, v217
	v_dual_fmac_f32 v72, v89, v217 :: v_dual_fmac_f32 v73, v99, v217
	v_dual_fmac_f32 v82, v109, v217 :: v_dual_fmac_f32 v83, v119, v217
	v_dual_fmac_f32 v52, v50, v218 :: v_dual_fmac_f32 v53, v60, v218
	v_dual_fmac_f32 v62, v70, v218 :: v_dual_fmac_f32 v63, v80, v218
	v_dual_fmac_f32 v72, v90, v218 :: v_dual_fmac_f32 v73, v100, v218
	v_dual_fmac_f32 v82, v110, v218 :: v_dual_fmac_f32 v83, v120, v218
	v_dual_fmac_f32 v52, v51, v219 :: v_dual_fmac_f32 v53, v61, v219
	v_dual_fmac_f32 v62, v71, v219 :: v_dual_fmac_f32 v63, v81, v219
	v_dual_fmac_f32 v72, v91, v219 :: v_dual_fmac_f32 v73, v101, v219
	v_dual_fmac_f32 v82, v111, v219 :: v_dual_fmac_f32 v83, v121, v219
	global_load_b128 v[212:215], v3, s[32:33]
	global_load_b128 v[216:219], v3, s[32:33] offset:16
	v_dual_add_f32 v180, v52, v180 :: v_dual_add_f32 v181, v181, v53
	v_dual_add_f32 v182, v62, v182 :: v_dual_add_f32 v183, v183, v63
	v_dual_add_f32 v184, v72, v184 :: v_dual_add_f32 v185, v185, v73
	v_dual_add_f32 v186, v82, v186 :: v_dual_add_f32 v187, v187, v83
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s2
	s_wait_loadcnt 0x0
	.Lrx_b3_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[196:199], v2, s[28:29]
	global_load_b128 v[200:203], v2, s[28:29] offset:16
	global_load_b128 v[204:207], v2, s[30:31]
	global_load_b128 v[208:211], v2, s[30:31] offset:16
	global_load_b128 v[212:215], v2, s[32:33]
	global_load_b128 v[216:219], v2, s[32:33] offset:16
	.Lrx_b3_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x6
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v44, v196 :: v_dual_mul_f32 v53, v55, v197
	v_dual_mul_f32 v62, v64, v196 :: v_dual_mul_f32 v63, v75, v197
	v_dual_mul_f32 v72, v84, v196 :: v_dual_mul_f32 v73, v95, v197
	v_dual_mul_f32 v82, v104, v196 :: v_dual_mul_f32 v83, v115, v197
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v92, v44, v204 :: v_dual_mul_f32 v93, v55, v205
	v_dual_mul_f32 v102, v64, v204 :: v_dual_mul_f32 v103, v75, v205
	v_dual_mul_f32 v112, v84, v204 :: v_dual_mul_f32 v113, v95, v205
	v_dual_mul_f32 v122, v104, v204 :: v_dual_mul_f32 v123, v115, v205
	v_dual_fmac_f32 v52, v45, v197 :: v_dual_fmac_f32 v53, v54, v196
	v_dual_fmac_f32 v62, v65, v197 :: v_dual_fmac_f32 v63, v74, v196
	v_dual_fmac_f32 v72, v85, v197 :: v_dual_fmac_f32 v73, v94, v196
	v_dual_fmac_f32 v82, v105, v197 :: v_dual_fmac_f32 v83, v114, v196
	v_dual_fmac_f32 v92, v45, v205 :: v_dual_fmac_f32 v93, v54, v204
	v_dual_fmac_f32 v102, v65, v205 :: v_dual_fmac_f32 v103, v74, v204
	v_dual_fmac_f32 v112, v85, v205 :: v_dual_fmac_f32 v113, v94, v204
	v_dual_fmac_f32 v122, v105, v205 :: v_dual_fmac_f32 v123, v114, v204
	v_dual_fmac_f32 v52, v46, v198 :: v_dual_fmac_f32 v53, v56, v198
	v_dual_fmac_f32 v62, v66, v198 :: v_dual_fmac_f32 v63, v76, v198
	v_dual_fmac_f32 v72, v86, v198 :: v_dual_fmac_f32 v73, v96, v198
	v_dual_fmac_f32 v82, v106, v198 :: v_dual_fmac_f32 v83, v116, v198
	v_dual_fmac_f32 v92, v46, v206 :: v_dual_fmac_f32 v93, v56, v206
	v_dual_fmac_f32 v102, v66, v206 :: v_dual_fmac_f32 v103, v76, v206
	v_dual_fmac_f32 v112, v86, v206 :: v_dual_fmac_f32 v113, v96, v206
	v_dual_fmac_f32 v122, v106, v206 :: v_dual_fmac_f32 v123, v116, v206
	v_dual_fmac_f32 v52, v47, v199 :: v_dual_fmac_f32 v53, v57, v199
	v_dual_fmac_f32 v62, v67, v199 :: v_dual_fmac_f32 v63, v77, v199
	v_dual_fmac_f32 v72, v87, v199 :: v_dual_fmac_f32 v73, v97, v199
	v_dual_fmac_f32 v82, v107, v199 :: v_dual_fmac_f32 v83, v117, v199
	v_dual_fmac_f32 v92, v47, v207 :: v_dual_fmac_f32 v93, v57, v207
	v_dual_fmac_f32 v102, v67, v207 :: v_dual_fmac_f32 v103, v77, v207
	v_dual_fmac_f32 v112, v87, v207 :: v_dual_fmac_f32 v113, v97, v207
	v_dual_fmac_f32 v122, v107, v207 :: v_dual_fmac_f32 v123, v117, v207
	v_dual_fmac_f32 v52, v48, v200 :: v_dual_fmac_f32 v53, v58, v200
	v_dual_fmac_f32 v62, v68, v200 :: v_dual_fmac_f32 v63, v78, v200
	v_dual_fmac_f32 v72, v88, v200 :: v_dual_fmac_f32 v73, v98, v200
	v_dual_fmac_f32 v82, v108, v200 :: v_dual_fmac_f32 v83, v118, v200
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v92, v48, v208 :: v_dual_fmac_f32 v93, v58, v208
	v_dual_fmac_f32 v102, v68, v208 :: v_dual_fmac_f32 v103, v78, v208
	v_dual_fmac_f32 v112, v88, v208 :: v_dual_fmac_f32 v113, v98, v208
	v_dual_fmac_f32 v122, v108, v208 :: v_dual_fmac_f32 v123, v118, v208
	v_dual_fmac_f32 v52, v49, v201 :: v_dual_fmac_f32 v53, v59, v201
	v_dual_fmac_f32 v62, v69, v201 :: v_dual_fmac_f32 v63, v79, v201
	v_dual_fmac_f32 v72, v89, v201 :: v_dual_fmac_f32 v73, v99, v201
	v_dual_fmac_f32 v82, v109, v201 :: v_dual_fmac_f32 v83, v119, v201
	v_dual_fmac_f32 v92, v49, v209 :: v_dual_fmac_f32 v93, v59, v209
	v_dual_fmac_f32 v102, v69, v209 :: v_dual_fmac_f32 v103, v79, v209
	v_dual_fmac_f32 v112, v89, v209 :: v_dual_fmac_f32 v113, v99, v209
	v_dual_fmac_f32 v122, v109, v209 :: v_dual_fmac_f32 v123, v119, v209
	v_dual_fmac_f32 v52, v50, v202 :: v_dual_fmac_f32 v53, v60, v202
	v_dual_fmac_f32 v62, v70, v202 :: v_dual_fmac_f32 v63, v80, v202
	v_dual_fmac_f32 v72, v90, v202 :: v_dual_fmac_f32 v73, v100, v202
	v_dual_fmac_f32 v82, v110, v202 :: v_dual_fmac_f32 v83, v120, v202
	v_dual_fmac_f32 v92, v50, v210 :: v_dual_fmac_f32 v93, v60, v210
	v_dual_fmac_f32 v102, v70, v210 :: v_dual_fmac_f32 v103, v80, v210
	v_dual_fmac_f32 v112, v90, v210 :: v_dual_fmac_f32 v113, v100, v210
	v_dual_fmac_f32 v122, v110, v210 :: v_dual_fmac_f32 v123, v120, v210
	v_dual_fmac_f32 v52, v51, v203 :: v_dual_fmac_f32 v53, v61, v203
	v_dual_fmac_f32 v62, v71, v203 :: v_dual_fmac_f32 v63, v81, v203
	v_dual_fmac_f32 v72, v91, v203 :: v_dual_fmac_f32 v73, v101, v203
	v_dual_fmac_f32 v82, v111, v203 :: v_dual_fmac_f32 v83, v121, v203
	v_dual_fmac_f32 v92, v51, v211 :: v_dual_fmac_f32 v93, v61, v211
	v_dual_fmac_f32 v102, v71, v211 :: v_dual_fmac_f32 v103, v81, v211
	v_dual_fmac_f32 v112, v91, v211 :: v_dual_fmac_f32 v113, v101, v211
	v_dual_fmac_f32 v122, v111, v211 :: v_dual_fmac_f32 v123, v121, v211
	global_load_b128 v[196:199], v3, s[28:29]
	global_load_b128 v[200:203], v3, s[28:29] offset:16
	global_load_b128 v[204:207], v3, s[30:31]
	global_load_b128 v[208:211], v3, s[30:31] offset:16
	v_dual_add_f32 v140, v52, v140 :: v_dual_add_f32 v141, v141, v53
	v_dual_add_f32 v142, v62, v142 :: v_dual_add_f32 v143, v143, v63
	v_dual_add_f32 v144, v72, v144 :: v_dual_add_f32 v145, v145, v73
	v_dual_add_f32 v146, v82, v146 :: v_dual_add_f32 v147, v147, v83
	v_dual_add_f32 v164, v92, v164 :: v_dual_add_f32 v165, v165, v93
	v_dual_add_f32 v166, v102, v166 :: v_dual_add_f32 v167, v167, v103
	v_dual_add_f32 v168, v112, v168 :: v_dual_add_f32 v169, v169, v113
	v_dual_add_f32 v170, v122, v170 :: v_dual_add_f32 v171, v171, v123
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v44, v212 :: v_dual_mul_f32 v53, v55, v213
	v_dual_mul_f32 v62, v64, v212 :: v_dual_mul_f32 v63, v75, v213
	v_dual_mul_f32 v72, v84, v212 :: v_dual_mul_f32 v73, v95, v213
	v_dual_mul_f32 v82, v104, v212 :: v_dual_mul_f32 v83, v115, v213
	v_dual_fmac_f32 v52, v45, v213 :: v_dual_fmac_f32 v53, v54, v212
	v_dual_fmac_f32 v62, v65, v213 :: v_dual_fmac_f32 v63, v74, v212
	v_dual_fmac_f32 v72, v85, v213 :: v_dual_fmac_f32 v73, v94, v212
	v_dual_fmac_f32 v82, v105, v213 :: v_dual_fmac_f32 v83, v114, v212
	v_dual_fmac_f32 v52, v46, v214 :: v_dual_fmac_f32 v53, v56, v214
	v_dual_fmac_f32 v62, v66, v214 :: v_dual_fmac_f32 v63, v76, v214
	v_dual_fmac_f32 v72, v86, v214 :: v_dual_fmac_f32 v73, v96, v214
	v_dual_fmac_f32 v82, v106, v214 :: v_dual_fmac_f32 v83, v116, v214
	v_dual_fmac_f32 v52, v47, v215 :: v_dual_fmac_f32 v53, v57, v215
	v_dual_fmac_f32 v62, v67, v215 :: v_dual_fmac_f32 v63, v77, v215
	v_dual_fmac_f32 v72, v87, v215 :: v_dual_fmac_f32 v73, v97, v215
	v_dual_fmac_f32 v82, v107, v215 :: v_dual_fmac_f32 v83, v117, v215
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v48, v216 :: v_dual_fmac_f32 v53, v58, v216
	v_dual_fmac_f32 v62, v68, v216 :: v_dual_fmac_f32 v63, v78, v216
	v_dual_fmac_f32 v72, v88, v216 :: v_dual_fmac_f32 v73, v98, v216
	v_dual_fmac_f32 v82, v108, v216 :: v_dual_fmac_f32 v83, v118, v216
	v_dual_fmac_f32 v52, v49, v217 :: v_dual_fmac_f32 v53, v59, v217
	v_dual_fmac_f32 v62, v69, v217 :: v_dual_fmac_f32 v63, v79, v217
	v_dual_fmac_f32 v72, v89, v217 :: v_dual_fmac_f32 v73, v99, v217
	v_dual_fmac_f32 v82, v109, v217 :: v_dual_fmac_f32 v83, v119, v217
	v_dual_fmac_f32 v52, v50, v218 :: v_dual_fmac_f32 v53, v60, v218
	v_dual_fmac_f32 v62, v70, v218 :: v_dual_fmac_f32 v63, v80, v218
	v_dual_fmac_f32 v72, v90, v218 :: v_dual_fmac_f32 v73, v100, v218
	v_dual_fmac_f32 v82, v110, v218 :: v_dual_fmac_f32 v83, v120, v218
	v_dual_fmac_f32 v52, v51, v219 :: v_dual_fmac_f32 v53, v61, v219
	v_dual_fmac_f32 v62, v71, v219 :: v_dual_fmac_f32 v63, v81, v219
	v_dual_fmac_f32 v72, v91, v219 :: v_dual_fmac_f32 v73, v101, v219
	v_dual_fmac_f32 v82, v111, v219 :: v_dual_fmac_f32 v83, v121, v219
	global_load_b128 v[212:215], v3, s[32:33]
	global_load_b128 v[216:219], v3, s[32:33] offset:16
	v_dual_add_f32 v188, v52, v188 :: v_dual_add_f32 v189, v189, v53
	v_dual_add_f32 v190, v62, v190 :: v_dual_add_f32 v191, v191, v63
	v_dual_add_f32 v192, v72, v192 :: v_dual_add_f32 v193, v193, v73
	v_dual_add_f32 v194, v82, v194 :: v_dual_add_f32 v195, v195, v83
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s3
	s_wait_loadcnt 0x0
	.Lrx_b3_s3_done:
	v_dual_add_f32 v132, v132, v140 :: v_dual_add_f32 v133, v133, v141
	v_dual_add_f32 v134, v134, v142 :: v_dual_add_f32 v135, v135, v143
	v_dual_add_f32 v136, v136, v144 :: v_dual_add_f32 v137, v137, v145
	v_dual_add_f32 v138, v138, v146 :: v_dual_add_f32 v139, v139, v147
	v_dual_add_f32 v156, v156, v164 :: v_dual_add_f32 v157, v157, v165
	v_dual_add_f32 v158, v158, v166 :: v_dual_add_f32 v159, v159, v167
	v_dual_add_f32 v160, v160, v168 :: v_dual_add_f32 v161, v161, v169
	v_dual_add_f32 v162, v162, v170 :: v_dual_add_f32 v163, v163, v171
	v_dual_add_f32 v180, v180, v188 :: v_dual_add_f32 v181, v181, v189
	v_dual_add_f32 v182, v182, v190 :: v_dual_add_f32 v183, v183, v191
	v_dual_add_f32 v184, v184, v192 :: v_dual_add_f32 v185, v185, v193
	v_dual_add_f32 v186, v186, v194 :: v_dual_add_f32 v187, v187, v195
	v_dual_add_f32 v124, v124, v132 :: v_dual_add_f32 v125, v125, v133
	v_dual_add_f32 v126, v126, v134 :: v_dual_add_f32 v127, v127, v135
	v_dual_add_f32 v128, v128, v136 :: v_dual_add_f32 v129, v129, v137
	v_dual_add_f32 v130, v130, v138 :: v_dual_add_f32 v131, v131, v139
	v_dual_add_f32 v148, v148, v156 :: v_dual_add_f32 v149, v149, v157
	v_dual_add_f32 v150, v150, v158 :: v_dual_add_f32 v151, v151, v159
	v_dual_add_f32 v152, v152, v160 :: v_dual_add_f32 v153, v153, v161
	v_dual_add_f32 v154, v154, v162 :: v_dual_add_f32 v155, v155, v163
	v_dual_add_f32 v172, v172, v180 :: v_dual_add_f32 v173, v173, v181
	v_dual_add_f32 v174, v174, v182 :: v_dual_add_f32 v175, v175, v183
	v_dual_add_f32 v176, v176, v184 :: v_dual_add_f32 v177, v177, v185
	v_dual_add_f32 v178, v178, v186 :: v_dual_add_f32 v179, v179, v187
	ds_swizzle_b32 v196, v124 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v197, v125 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v198, v126 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v199, v127 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v200, v128 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v201, v129 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v202, v130 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v203, v131 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v204, v148 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v205, v149 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v206, v150 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v207, v151 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v208, v152 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v209, v153 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v210, v154 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v211, v155 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v212, v172 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v213, v173 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v214, v174 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v215, v175 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v216, v176 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v217, v177 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v218, v178 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v219, v179 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x17
	v_add_f32_e32 v124, v124, v196
	s_wait_dscnt 0x16
	v_add_f32_e32 v125, v125, v197
	s_wait_dscnt 0x15
	v_add_f32_e32 v126, v126, v198
	s_wait_dscnt 0x14
	v_add_f32_e32 v127, v127, v199
	s_wait_dscnt 0x13
	v_add_f32_e32 v128, v128, v200
	s_wait_dscnt 0x12
	v_add_f32_e32 v129, v129, v201
	s_wait_dscnt 0x11
	v_add_f32_e32 v130, v130, v202
	s_wait_dscnt 0x10
	v_add_f32_e32 v131, v131, v203
	s_wait_dscnt 0xf
	v_add_f32_e32 v148, v148, v204
	s_wait_dscnt 0xe
	v_add_f32_e32 v149, v149, v205
	s_wait_dscnt 0xd
	v_add_f32_e32 v150, v150, v206
	s_wait_dscnt 0xc
	v_add_f32_e32 v151, v151, v207
	s_wait_dscnt 0xb
	v_add_f32_e32 v152, v152, v208
	s_wait_dscnt 0xa
	v_add_f32_e32 v153, v153, v209
	s_wait_dscnt 0x9
	v_add_f32_e32 v154, v154, v210
	s_wait_dscnt 0x8
	v_add_f32_e32 v155, v155, v211
	s_wait_dscnt 0x7
	v_add_f32_e32 v172, v172, v212
	s_wait_dscnt 0x6
	v_add_f32_e32 v173, v173, v213
	s_wait_dscnt 0x5
	v_add_f32_e32 v174, v174, v214
	s_wait_dscnt 0x4
	v_add_f32_e32 v175, v175, v215
	s_wait_dscnt 0x3
	v_add_f32_e32 v176, v176, v216
	s_wait_dscnt 0x2
	v_add_f32_e32 v177, v177, v217
	s_wait_dscnt 0x1
	v_add_f32_e32 v178, v178, v218
	s_wait_dscnt 0x0
	v_add_f32_e32 v179, v179, v219
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v196, v3, v124
	ds_bpermute_b32 v197, v3, v125
	ds_bpermute_b32 v198, v3, v126
	ds_bpermute_b32 v199, v3, v127
	ds_bpermute_b32 v200, v3, v128
	ds_bpermute_b32 v201, v3, v129
	ds_bpermute_b32 v202, v3, v130
	ds_bpermute_b32 v203, v3, v131
	ds_bpermute_b32 v204, v3, v148
	ds_bpermute_b32 v205, v3, v149
	ds_bpermute_b32 v206, v3, v150
	ds_bpermute_b32 v207, v3, v151
	ds_bpermute_b32 v208, v3, v152
	ds_bpermute_b32 v209, v3, v153
	ds_bpermute_b32 v210, v3, v154
	ds_bpermute_b32 v211, v3, v155
	ds_bpermute_b32 v212, v3, v172
	ds_bpermute_b32 v213, v3, v173
	ds_bpermute_b32 v214, v3, v174
	ds_bpermute_b32 v215, v3, v175
	ds_bpermute_b32 v216, v3, v176
	ds_bpermute_b32 v217, v3, v177
	ds_bpermute_b32 v218, v3, v178
	ds_bpermute_b32 v219, v3, v179
	s_wait_dscnt 0x17
	v_add_f32_e32 v124, v124, v196
	s_wait_dscnt 0x16
	v_add_f32_e32 v125, v125, v197
	s_wait_dscnt 0x15
	v_add_f32_e32 v126, v126, v198
	s_wait_dscnt 0x14
	v_add_f32_e32 v127, v127, v199
	s_wait_dscnt 0x13
	v_add_f32_e32 v128, v128, v200
	s_wait_dscnt 0x12
	v_add_f32_e32 v129, v129, v201
	s_wait_dscnt 0x11
	v_add_f32_e32 v130, v130, v202
	s_wait_dscnt 0x10
	v_add_f32_e32 v131, v131, v203
	s_wait_dscnt 0xf
	v_add_f32_e32 v148, v148, v204
	s_wait_dscnt 0xe
	v_add_f32_e32 v149, v149, v205
	s_wait_dscnt 0xd
	v_add_f32_e32 v150, v150, v206
	s_wait_dscnt 0xc
	v_add_f32_e32 v151, v151, v207
	s_wait_dscnt 0xb
	v_add_f32_e32 v152, v152, v208
	s_wait_dscnt 0xa
	v_add_f32_e32 v153, v153, v209
	s_wait_dscnt 0x9
	v_add_f32_e32 v154, v154, v210
	s_wait_dscnt 0x8
	v_add_f32_e32 v155, v155, v211
	s_wait_dscnt 0x7
	v_add_f32_e32 v172, v172, v212
	s_wait_dscnt 0x6
	v_add_f32_e32 v173, v173, v213
	s_wait_dscnt 0x5
	v_add_f32_e32 v174, v174, v214
	s_wait_dscnt 0x4
	v_add_f32_e32 v175, v175, v215
	s_wait_dscnt 0x3
	v_add_f32_e32 v176, v176, v216
	s_wait_dscnt 0x2
	v_add_f32_e32 v177, v177, v217
	s_wait_dscnt 0x1
	v_add_f32_e32 v178, v178, v218
	s_wait_dscnt 0x0
	v_add_f32_e32 v179, v179, v219
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v196, v3, v124
	ds_bpermute_b32 v197, v3, v125
	ds_bpermute_b32 v198, v3, v126
	ds_bpermute_b32 v199, v3, v127
	ds_bpermute_b32 v200, v3, v128
	ds_bpermute_b32 v201, v3, v129
	ds_bpermute_b32 v202, v3, v130
	ds_bpermute_b32 v203, v3, v131
	ds_bpermute_b32 v204, v3, v148
	ds_bpermute_b32 v205, v3, v149
	ds_bpermute_b32 v206, v3, v150
	ds_bpermute_b32 v207, v3, v151
	ds_bpermute_b32 v208, v3, v152
	ds_bpermute_b32 v209, v3, v153
	ds_bpermute_b32 v210, v3, v154
	ds_bpermute_b32 v211, v3, v155
	ds_bpermute_b32 v212, v3, v172
	ds_bpermute_b32 v213, v3, v173
	ds_bpermute_b32 v214, v3, v174
	ds_bpermute_b32 v215, v3, v175
	ds_bpermute_b32 v216, v3, v176
	ds_bpermute_b32 v217, v3, v177
	ds_bpermute_b32 v218, v3, v178
	ds_bpermute_b32 v219, v3, v179
	s_wait_dscnt 0x17
	v_add_f32_e32 v124, v124, v196
	s_wait_dscnt 0x16
	v_add_f32_e32 v125, v125, v197
	s_wait_dscnt 0x15
	v_add_f32_e32 v126, v126, v198
	s_wait_dscnt 0x14
	v_add_f32_e32 v127, v127, v199
	s_wait_dscnt 0x13
	v_add_f32_e32 v128, v128, v200
	s_wait_dscnt 0x12
	v_add_f32_e32 v129, v129, v201
	s_wait_dscnt 0x11
	v_add_f32_e32 v130, v130, v202
	s_wait_dscnt 0x10
	v_add_f32_e32 v131, v131, v203
	s_wait_dscnt 0xf
	v_add_f32_e32 v148, v148, v204
	s_wait_dscnt 0xe
	v_add_f32_e32 v149, v149, v205
	s_wait_dscnt 0xd
	v_add_f32_e32 v150, v150, v206
	s_wait_dscnt 0xc
	v_add_f32_e32 v151, v151, v207
	s_wait_dscnt 0xb
	v_add_f32_e32 v152, v152, v208
	s_wait_dscnt 0xa
	v_add_f32_e32 v153, v153, v209
	s_wait_dscnt 0x9
	v_add_f32_e32 v154, v154, v210
	s_wait_dscnt 0x8
	v_add_f32_e32 v155, v155, v211
	s_wait_dscnt 0x7
	v_add_f32_e32 v172, v172, v212
	s_wait_dscnt 0x6
	v_add_f32_e32 v173, v173, v213
	s_wait_dscnt 0x5
	v_add_f32_e32 v174, v174, v214
	s_wait_dscnt 0x4
	v_add_f32_e32 v175, v175, v215
	s_wait_dscnt 0x3
	v_add_f32_e32 v176, v176, v216
	s_wait_dscnt 0x2
	v_add_f32_e32 v177, v177, v217
	s_wait_dscnt 0x1
	v_add_f32_e32 v178, v178, v218
	s_wait_dscnt 0x0
	v_add_f32_e32 v179, v179, v219
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v196, v3, v124
	ds_bpermute_b32 v197, v3, v125
	ds_bpermute_b32 v198, v3, v126
	ds_bpermute_b32 v199, v3, v127
	ds_bpermute_b32 v200, v3, v128
	ds_bpermute_b32 v201, v3, v129
	ds_bpermute_b32 v202, v3, v130
	ds_bpermute_b32 v203, v3, v131
	ds_bpermute_b32 v204, v3, v148
	ds_bpermute_b32 v205, v3, v149
	ds_bpermute_b32 v206, v3, v150
	ds_bpermute_b32 v207, v3, v151
	ds_bpermute_b32 v208, v3, v152
	ds_bpermute_b32 v209, v3, v153
	ds_bpermute_b32 v210, v3, v154
	ds_bpermute_b32 v211, v3, v155
	ds_bpermute_b32 v212, v3, v172
	ds_bpermute_b32 v213, v3, v173
	ds_bpermute_b32 v214, v3, v174
	ds_bpermute_b32 v215, v3, v175
	ds_bpermute_b32 v216, v3, v176
	ds_bpermute_b32 v217, v3, v177
	ds_bpermute_b32 v218, v3, v178
	ds_bpermute_b32 v219, v3, v179
	s_wait_dscnt 0x17
	v_add_f32_e32 v124, v124, v196
	s_wait_dscnt 0x16
	v_add_f32_e32 v125, v125, v197
	s_wait_dscnt 0x15
	v_add_f32_e32 v126, v126, v198
	s_wait_dscnt 0x14
	v_add_f32_e32 v127, v127, v199
	s_wait_dscnt 0x13
	v_add_f32_e32 v128, v128, v200
	s_wait_dscnt 0x12
	v_add_f32_e32 v129, v129, v201
	s_wait_dscnt 0x11
	v_add_f32_e32 v130, v130, v202
	s_wait_dscnt 0x10
	v_add_f32_e32 v131, v131, v203
	s_wait_dscnt 0xf
	v_add_f32_e32 v148, v148, v204
	s_wait_dscnt 0xe
	v_add_f32_e32 v149, v149, v205
	s_wait_dscnt 0xd
	v_add_f32_e32 v150, v150, v206
	s_wait_dscnt 0xc
	v_add_f32_e32 v151, v151, v207
	s_wait_dscnt 0xb
	v_add_f32_e32 v152, v152, v208
	s_wait_dscnt 0xa
	v_add_f32_e32 v153, v153, v209
	s_wait_dscnt 0x9
	v_add_f32_e32 v154, v154, v210
	s_wait_dscnt 0x8
	v_add_f32_e32 v155, v155, v211
	s_wait_dscnt 0x7
	v_add_f32_e32 v172, v172, v212
	s_wait_dscnt 0x6
	v_add_f32_e32 v173, v173, v213
	s_wait_dscnt 0x5
	v_add_f32_e32 v174, v174, v214
	s_wait_dscnt 0x4
	v_add_f32_e32 v175, v175, v215
	s_wait_dscnt 0x3
	v_add_f32_e32 v176, v176, v216
	s_wait_dscnt 0x2
	v_add_f32_e32 v177, v177, v217
	s_wait_dscnt 0x1
	v_add_f32_e32 v178, v178, v218
	s_wait_dscnt 0x0
	v_add_f32_e32 v179, v179, v219
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v196, v3, v124
	ds_bpermute_b32 v197, v3, v125
	ds_bpermute_b32 v198, v3, v126
	ds_bpermute_b32 v199, v3, v127
	ds_bpermute_b32 v200, v3, v128
	ds_bpermute_b32 v201, v3, v129
	ds_bpermute_b32 v202, v3, v130
	ds_bpermute_b32 v203, v3, v131
	ds_bpermute_b32 v204, v3, v148
	ds_bpermute_b32 v205, v3, v149
	ds_bpermute_b32 v206, v3, v150
	ds_bpermute_b32 v207, v3, v151
	ds_bpermute_b32 v208, v3, v152
	ds_bpermute_b32 v209, v3, v153
	ds_bpermute_b32 v210, v3, v154
	ds_bpermute_b32 v211, v3, v155
	ds_bpermute_b32 v212, v3, v172
	ds_bpermute_b32 v213, v3, v173
	ds_bpermute_b32 v214, v3, v174
	ds_bpermute_b32 v215, v3, v175
	ds_bpermute_b32 v216, v3, v176
	ds_bpermute_b32 v217, v3, v177
	ds_bpermute_b32 v218, v3, v178
	ds_bpermute_b32 v219, v3, v179
	s_wait_dscnt 0x17
	v_add_f32_e32 v124, v124, v196
	s_wait_dscnt 0x16
	v_add_f32_e32 v125, v125, v197
	s_wait_dscnt 0x15
	v_add_f32_e32 v126, v126, v198
	s_wait_dscnt 0x14
	v_add_f32_e32 v127, v127, v199
	s_wait_dscnt 0x13
	v_add_f32_e32 v128, v128, v200
	s_wait_dscnt 0x12
	v_add_f32_e32 v129, v129, v201
	s_wait_dscnt 0x11
	v_add_f32_e32 v130, v130, v202
	s_wait_dscnt 0x10
	v_add_f32_e32 v131, v131, v203
	s_wait_dscnt 0xf
	v_add_f32_e32 v148, v148, v204
	s_wait_dscnt 0xe
	v_add_f32_e32 v149, v149, v205
	s_wait_dscnt 0xd
	v_add_f32_e32 v150, v150, v206
	s_wait_dscnt 0xc
	v_add_f32_e32 v151, v151, v207
	s_wait_dscnt 0xb
	v_add_f32_e32 v152, v152, v208
	s_wait_dscnt 0xa
	v_add_f32_e32 v153, v153, v209
	s_wait_dscnt 0x9
	v_add_f32_e32 v154, v154, v210
	s_wait_dscnt 0x8
	v_add_f32_e32 v155, v155, v211
	s_wait_dscnt 0x7
	v_add_f32_e32 v172, v172, v212
	s_wait_dscnt 0x6
	v_add_f32_e32 v173, v173, v213
	s_wait_dscnt 0x5
	v_add_f32_e32 v174, v174, v214
	s_wait_dscnt 0x4
	v_add_f32_e32 v175, v175, v215
	s_wait_dscnt 0x3
	v_add_f32_e32 v176, v176, v216
	s_wait_dscnt 0x2
	v_add_f32_e32 v177, v177, v217
	s_wait_dscnt 0x1
	v_add_f32_e32 v178, v178, v218
	s_wait_dscnt 0x0
	v_add_f32_e32 v179, v179, v219
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v68, s46
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v69, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v70, s47
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b3_p0_single
	global_load_b32 v44, v68, s[8:9] offset:0
	global_load_b32 v45, v68, s[8:9] offset:4
	global_load_b32 v52, v69, s[8:9] offset:0
	global_load_b32 v53, v69, s[8:9] offset:4
	global_load_b32 v60, v70, s[8:9] offset:0
	global_load_b32 v61, v70, s[8:9] offset:4
	s_wait_loadcnt 0x5
	v_add_f32_e32 v44, v124, v44
	global_store_b32 v68, v44, s[8:9] offset:0
	s_wait_loadcnt 0x4
	v_add_f32_e32 v45, v125, v45
	global_store_b32 v68, v45, s[8:9] offset:4
	s_wait_loadcnt 0x3
	v_add_f32_e32 v52, v148, v52
	global_store_b32 v69, v52, s[8:9] offset:0
	s_wait_loadcnt 0x2
	v_add_f32_e32 v53, v149, v53
	global_store_b32 v69, v53, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v60, v172, v60
	global_store_b32 v70, v60, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v61, v173, v61
	global_store_b32 v70, v61, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b3_p0_next
	.Lrx_b3_p0_single:
	global_load_b32 v44, v68, s[8:9] offset:0
	global_load_b32 v52, v69, s[8:9] offset:0
	global_load_b32 v60, v70, s[8:9] offset:0
	s_wait_loadcnt 0x2
	v_add_f32_e32 v44, v44, v124
	global_store_b32 v68, v44, s[8:9] offset:0
	s_wait_loadcnt 0x1
	v_add_f32_e32 v52, v52, v148
	global_store_b32 v69, v52, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v60, v60, v172
	global_store_b32 v70, v60, s[8:9] offset:0
	s_wait_storecnt 0x0
	s_branch .Lrx_b3_stored
	.Lrx_b3_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b3_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b3_p1_single
	global_load_b32 v46, v68, s[8:9] offset:8
	global_load_b32 v47, v68, s[8:9] offset:12
	global_load_b32 v54, v69, s[8:9] offset:8
	global_load_b32 v55, v69, s[8:9] offset:12
	global_load_b32 v62, v70, s[8:9] offset:8
	global_load_b32 v63, v70, s[8:9] offset:12
	s_wait_loadcnt 0x5
	v_add_f32_e32 v46, v126, v46
	global_store_b32 v68, v46, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v47, v127, v47
	global_store_b32 v68, v47, s[8:9] offset:12
	s_wait_loadcnt 0x3
	v_add_f32_e32 v54, v150, v54
	global_store_b32 v69, v54, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v55, v151, v55
	global_store_b32 v69, v55, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v62, v174, v62
	global_store_b32 v70, v62, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v63, v175, v63
	global_store_b32 v70, v63, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b3_p1_next
	.Lrx_b3_p1_single:
	global_load_b32 v46, v68, s[8:9] offset:8
	global_load_b32 v54, v69, s[8:9] offset:8
	global_load_b32 v62, v70, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v46, v46, v126
	global_store_b32 v68, v46, s[8:9] offset:8
	s_wait_loadcnt 0x1
	v_add_f32_e32 v54, v54, v150
	global_store_b32 v69, v54, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v62, v62, v174
	global_store_b32 v70, v62, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b3_stored
	.Lrx_b3_p1_next:
	s_add_co_i32 s13, s12, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b3_stored
	s_add_co_i32 s13, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b3_p2_single
	global_load_b32 v48, v68, s[8:9] offset:16
	global_load_b32 v49, v68, s[8:9] offset:20
	global_load_b32 v56, v69, s[8:9] offset:16
	global_load_b32 v57, v69, s[8:9] offset:20
	global_load_b32 v64, v70, s[8:9] offset:16
	global_load_b32 v65, v70, s[8:9] offset:20
	s_wait_loadcnt 0x5
	v_add_f32_e32 v48, v128, v48
	global_store_b32 v68, v48, s[8:9] offset:16
	s_wait_loadcnt 0x4
	v_add_f32_e32 v49, v129, v49
	global_store_b32 v68, v49, s[8:9] offset:20
	s_wait_loadcnt 0x3
	v_add_f32_e32 v56, v152, v56
	global_store_b32 v69, v56, s[8:9] offset:16
	s_wait_loadcnt 0x2
	v_add_f32_e32 v57, v153, v57
	global_store_b32 v69, v57, s[8:9] offset:20
	s_wait_loadcnt 0x1
	v_add_f32_e32 v64, v176, v64
	global_store_b32 v70, v64, s[8:9] offset:16
	s_wait_loadcnt 0x0
	v_add_f32_e32 v65, v177, v65
	global_store_b32 v70, v65, s[8:9] offset:20
	s_wait_storecnt 0x0
	s_branch .Lrx_b3_p2_next
	.Lrx_b3_p2_single:
	global_load_b32 v48, v68, s[8:9] offset:16
	global_load_b32 v56, v69, s[8:9] offset:16
	global_load_b32 v64, v70, s[8:9] offset:16
	s_wait_loadcnt 0x2
	v_add_f32_e32 v48, v48, v128
	global_store_b32 v68, v48, s[8:9] offset:16
	s_wait_loadcnt 0x1
	v_add_f32_e32 v56, v56, v152
	global_store_b32 v69, v56, s[8:9] offset:16
	s_wait_loadcnt 0x0
	v_add_f32_e32 v64, v64, v176
	global_store_b32 v70, v64, s[8:9] offset:16
	s_wait_storecnt 0x0
	s_branch .Lrx_b3_stored
	.Lrx_b3_p2_next:
	s_add_co_i32 s13, s12, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b3_stored
	s_add_co_i32 s13, s12, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b3_p3_single
	global_load_b32 v50, v68, s[8:9] offset:24
	global_load_b32 v51, v68, s[8:9] offset:28
	global_load_b32 v58, v69, s[8:9] offset:24
	global_load_b32 v59, v69, s[8:9] offset:28
	global_load_b32 v66, v70, s[8:9] offset:24
	global_load_b32 v67, v70, s[8:9] offset:28
	s_wait_loadcnt 0x5
	v_add_f32_e32 v50, v130, v50
	global_store_b32 v68, v50, s[8:9] offset:24
	s_wait_loadcnt 0x4
	v_add_f32_e32 v51, v131, v51
	global_store_b32 v68, v51, s[8:9] offset:28
	s_wait_loadcnt 0x3
	v_add_f32_e32 v58, v154, v58
	global_store_b32 v69, v58, s[8:9] offset:24
	s_wait_loadcnt 0x2
	v_add_f32_e32 v59, v155, v59
	global_store_b32 v69, v59, s[8:9] offset:28
	s_wait_loadcnt 0x1
	v_add_f32_e32 v66, v178, v66
	global_store_b32 v70, v66, s[8:9] offset:24
	s_wait_loadcnt 0x0
	v_add_f32_e32 v67, v179, v67
	global_store_b32 v70, v67, s[8:9] offset:28
	s_wait_storecnt 0x0
	s_branch .Lrx_b3_stored
	.Lrx_b3_p3_single:
	global_load_b32 v50, v68, s[8:9] offset:24
	global_load_b32 v58, v69, s[8:9] offset:24
	global_load_b32 v66, v70, s[8:9] offset:24
	s_wait_loadcnt 0x2
	v_add_f32_e32 v50, v50, v130
	global_store_b32 v68, v50, s[8:9] offset:24
	s_wait_loadcnt 0x1
	v_add_f32_e32 v58, v58, v154
	global_store_b32 v69, v58, s[8:9] offset:24
	s_wait_loadcnt 0x0
	v_add_f32_e32 v66, v66, v178
	global_store_b32 v70, v66, s[8:9] offset:24
	s_wait_storecnt 0x0
	s_branch .Lrx_b3_stored
	.Lrx_b3_stored:
	s_branch .Lrx_end
	.Lrx_b4:
	s_mul_i32 s12, ttmp9, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v12, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v13, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v14, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v15, s13
	v_add_nc_u32_e32 v7, s13, v1
	s_add_co_i32 s13, s12, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v16, s13
	v_add_nc_u32_e32 v8, s13, v1
	s_add_co_i32 s13, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v17, s13
	v_add_nc_u32_e32 v9, s13, v1
	s_add_co_i32 s13, s12, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v18, s13
	v_add_nc_u32_e32 v10, s13, v1
	s_add_co_i32 s13, s12, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v19, s13
	v_add_nc_u32_e32 v11, s13, v1
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s0_done
	s_mov_b32 s15, 0x0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[220:223], v2, s[28:29]
	global_load_b128 v[224:227], v2, s[28:29] offset:16
	global_load_b128 v[228:231], v2, s[30:31]
	global_load_b128 v[232:235], v2, s[30:31] offset:16
	global_load_b128 v[236:239], v2, s[32:33]
	global_load_b128 v[240:243], v2, s[32:33] offset:16
	global_load_b128 v[244:247], v2, s[34:35]
	global_load_b128 v[248:251], v2, s[34:35] offset:16
	.Lrx_b4_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x16
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v52, v44, v220 :: v_dual_mul_f32 v53, v55, v221
	v_dual_mul_f32 v62, v64, v220 :: v_dual_mul_f32 v63, v75, v221
	v_dual_mul_f32 v72, v84, v220 :: v_dual_mul_f32 v73, v95, v221
	v_dual_mul_f32 v82, v104, v220 :: v_dual_mul_f32 v83, v115, v221
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v92, v44, v228 :: v_dual_mul_f32 v93, v55, v229
	v_dual_mul_f32 v102, v64, v228 :: v_dual_mul_f32 v103, v75, v229
	v_dual_mul_f32 v112, v84, v228 :: v_dual_mul_f32 v113, v95, v229
	v_dual_mul_f32 v122, v104, v228 :: v_dual_mul_f32 v123, v115, v229
	v_dual_fmac_f32 v52, v45, v221 :: v_dual_fmac_f32 v53, v54, v220
	v_dual_fmac_f32 v62, v65, v221 :: v_dual_fmac_f32 v63, v74, v220
	v_dual_fmac_f32 v72, v85, v221 :: v_dual_fmac_f32 v73, v94, v220
	v_dual_fmac_f32 v82, v105, v221 :: v_dual_fmac_f32 v83, v114, v220
	v_dual_fmac_f32 v92, v45, v229 :: v_dual_fmac_f32 v93, v54, v228
	v_dual_fmac_f32 v102, v65, v229 :: v_dual_fmac_f32 v103, v74, v228
	v_dual_fmac_f32 v112, v85, v229 :: v_dual_fmac_f32 v113, v94, v228
	v_dual_fmac_f32 v122, v105, v229 :: v_dual_fmac_f32 v123, v114, v228
	v_dual_fmac_f32 v52, v46, v222 :: v_dual_fmac_f32 v53, v56, v222
	v_dual_fmac_f32 v62, v66, v222 :: v_dual_fmac_f32 v63, v76, v222
	v_dual_fmac_f32 v72, v86, v222 :: v_dual_fmac_f32 v73, v96, v222
	v_dual_fmac_f32 v82, v106, v222 :: v_dual_fmac_f32 v83, v116, v222
	v_dual_fmac_f32 v92, v46, v230 :: v_dual_fmac_f32 v93, v56, v230
	v_dual_fmac_f32 v102, v66, v230 :: v_dual_fmac_f32 v103, v76, v230
	v_dual_fmac_f32 v112, v86, v230 :: v_dual_fmac_f32 v113, v96, v230
	v_dual_fmac_f32 v122, v106, v230 :: v_dual_fmac_f32 v123, v116, v230
	v_dual_fmac_f32 v52, v47, v223 :: v_dual_fmac_f32 v53, v57, v223
	v_dual_fmac_f32 v62, v67, v223 :: v_dual_fmac_f32 v63, v77, v223
	v_dual_fmac_f32 v72, v87, v223 :: v_dual_fmac_f32 v73, v97, v223
	v_dual_fmac_f32 v82, v107, v223 :: v_dual_fmac_f32 v83, v117, v223
	v_dual_fmac_f32 v92, v47, v231 :: v_dual_fmac_f32 v93, v57, v231
	v_dual_fmac_f32 v102, v67, v231 :: v_dual_fmac_f32 v103, v77, v231
	v_dual_fmac_f32 v112, v87, v231 :: v_dual_fmac_f32 v113, v97, v231
	v_dual_fmac_f32 v122, v107, v231 :: v_dual_fmac_f32 v123, v117, v231
	v_dual_fmac_f32 v52, v48, v224 :: v_dual_fmac_f32 v53, v58, v224
	v_dual_fmac_f32 v62, v68, v224 :: v_dual_fmac_f32 v63, v78, v224
	v_dual_fmac_f32 v72, v88, v224 :: v_dual_fmac_f32 v73, v98, v224
	v_dual_fmac_f32 v82, v108, v224 :: v_dual_fmac_f32 v83, v118, v224
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v92, v48, v232 :: v_dual_fmac_f32 v93, v58, v232
	v_dual_fmac_f32 v102, v68, v232 :: v_dual_fmac_f32 v103, v78, v232
	v_dual_fmac_f32 v112, v88, v232 :: v_dual_fmac_f32 v113, v98, v232
	v_dual_fmac_f32 v122, v108, v232 :: v_dual_fmac_f32 v123, v118, v232
	v_dual_fmac_f32 v52, v49, v225 :: v_dual_fmac_f32 v53, v59, v225
	v_dual_fmac_f32 v62, v69, v225 :: v_dual_fmac_f32 v63, v79, v225
	v_dual_fmac_f32 v72, v89, v225 :: v_dual_fmac_f32 v73, v99, v225
	v_dual_fmac_f32 v82, v109, v225 :: v_dual_fmac_f32 v83, v119, v225
	v_dual_fmac_f32 v92, v49, v233 :: v_dual_fmac_f32 v93, v59, v233
	v_dual_fmac_f32 v102, v69, v233 :: v_dual_fmac_f32 v103, v79, v233
	v_dual_fmac_f32 v112, v89, v233 :: v_dual_fmac_f32 v113, v99, v233
	v_dual_fmac_f32 v122, v109, v233 :: v_dual_fmac_f32 v123, v119, v233
	v_dual_fmac_f32 v52, v50, v226 :: v_dual_fmac_f32 v53, v60, v226
	v_dual_fmac_f32 v62, v70, v226 :: v_dual_fmac_f32 v63, v80, v226
	v_dual_fmac_f32 v72, v90, v226 :: v_dual_fmac_f32 v73, v100, v226
	v_dual_fmac_f32 v82, v110, v226 :: v_dual_fmac_f32 v83, v120, v226
	v_dual_fmac_f32 v92, v50, v234 :: v_dual_fmac_f32 v93, v60, v234
	v_dual_fmac_f32 v102, v70, v234 :: v_dual_fmac_f32 v103, v80, v234
	v_dual_fmac_f32 v112, v90, v234 :: v_dual_fmac_f32 v113, v100, v234
	v_dual_fmac_f32 v122, v110, v234 :: v_dual_fmac_f32 v123, v120, v234
	v_dual_fmac_f32 v52, v51, v227 :: v_dual_fmac_f32 v53, v61, v227
	v_dual_fmac_f32 v62, v71, v227 :: v_dual_fmac_f32 v63, v81, v227
	v_dual_fmac_f32 v72, v91, v227 :: v_dual_fmac_f32 v73, v101, v227
	v_dual_fmac_f32 v82, v111, v227 :: v_dual_fmac_f32 v83, v121, v227
	v_dual_fmac_f32 v92, v51, v235 :: v_dual_fmac_f32 v93, v61, v235
	v_dual_fmac_f32 v102, v71, v235 :: v_dual_fmac_f32 v103, v81, v235
	v_dual_fmac_f32 v112, v91, v235 :: v_dual_fmac_f32 v113, v101, v235
	v_dual_fmac_f32 v122, v111, v235 :: v_dual_fmac_f32 v123, v121, v235
	global_load_b128 v[220:223], v3, s[28:29]
	global_load_b128 v[224:227], v3, s[28:29] offset:16
	global_load_b128 v[228:231], v3, s[30:31]
	global_load_b128 v[232:235], v3, s[30:31] offset:16
	v_dual_add_f32 v124, v52, v124 :: v_dual_add_f32 v125, v125, v53
	v_dual_add_f32 v126, v62, v126 :: v_dual_add_f32 v127, v127, v63
	v_dual_add_f32 v128, v72, v128 :: v_dual_add_f32 v129, v129, v73
	v_dual_add_f32 v130, v82, v130 :: v_dual_add_f32 v131, v131, v83
	v_dual_add_f32 v148, v92, v148 :: v_dual_add_f32 v149, v149, v93
	v_dual_add_f32 v150, v102, v150 :: v_dual_add_f32 v151, v151, v103
	v_dual_add_f32 v152, v112, v152 :: v_dual_add_f32 v153, v153, v113
	v_dual_add_f32 v154, v122, v154 :: v_dual_add_f32 v155, v155, v123
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v52, v44, v236 :: v_dual_mul_f32 v53, v55, v237
	v_dual_mul_f32 v62, v64, v236 :: v_dual_mul_f32 v63, v75, v237
	v_dual_mul_f32 v72, v84, v236 :: v_dual_mul_f32 v73, v95, v237
	v_dual_mul_f32 v82, v104, v236 :: v_dual_mul_f32 v83, v115, v237
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v92, v44, v244 :: v_dual_mul_f32 v93, v55, v245
	v_dual_mul_f32 v102, v64, v244 :: v_dual_mul_f32 v103, v75, v245
	v_dual_mul_f32 v112, v84, v244 :: v_dual_mul_f32 v113, v95, v245
	v_dual_mul_f32 v122, v104, v244 :: v_dual_mul_f32 v123, v115, v245
	v_dual_fmac_f32 v52, v45, v237 :: v_dual_fmac_f32 v53, v54, v236
	v_dual_fmac_f32 v62, v65, v237 :: v_dual_fmac_f32 v63, v74, v236
	v_dual_fmac_f32 v72, v85, v237 :: v_dual_fmac_f32 v73, v94, v236
	v_dual_fmac_f32 v82, v105, v237 :: v_dual_fmac_f32 v83, v114, v236
	v_dual_fmac_f32 v92, v45, v245 :: v_dual_fmac_f32 v93, v54, v244
	v_dual_fmac_f32 v102, v65, v245 :: v_dual_fmac_f32 v103, v74, v244
	v_dual_fmac_f32 v112, v85, v245 :: v_dual_fmac_f32 v113, v94, v244
	v_dual_fmac_f32 v122, v105, v245 :: v_dual_fmac_f32 v123, v114, v244
	v_dual_fmac_f32 v52, v46, v238 :: v_dual_fmac_f32 v53, v56, v238
	v_dual_fmac_f32 v62, v66, v238 :: v_dual_fmac_f32 v63, v76, v238
	v_dual_fmac_f32 v72, v86, v238 :: v_dual_fmac_f32 v73, v96, v238
	v_dual_fmac_f32 v82, v106, v238 :: v_dual_fmac_f32 v83, v116, v238
	v_dual_fmac_f32 v92, v46, v246 :: v_dual_fmac_f32 v93, v56, v246
	v_dual_fmac_f32 v102, v66, v246 :: v_dual_fmac_f32 v103, v76, v246
	v_dual_fmac_f32 v112, v86, v246 :: v_dual_fmac_f32 v113, v96, v246
	v_dual_fmac_f32 v122, v106, v246 :: v_dual_fmac_f32 v123, v116, v246
	v_dual_fmac_f32 v52, v47, v239 :: v_dual_fmac_f32 v53, v57, v239
	v_dual_fmac_f32 v62, v67, v239 :: v_dual_fmac_f32 v63, v77, v239
	v_dual_fmac_f32 v72, v87, v239 :: v_dual_fmac_f32 v73, v97, v239
	v_dual_fmac_f32 v82, v107, v239 :: v_dual_fmac_f32 v83, v117, v239
	v_dual_fmac_f32 v92, v47, v247 :: v_dual_fmac_f32 v93, v57, v247
	v_dual_fmac_f32 v102, v67, v247 :: v_dual_fmac_f32 v103, v77, v247
	v_dual_fmac_f32 v112, v87, v247 :: v_dual_fmac_f32 v113, v97, v247
	v_dual_fmac_f32 v122, v107, v247 :: v_dual_fmac_f32 v123, v117, v247
	v_dual_fmac_f32 v52, v48, v240 :: v_dual_fmac_f32 v53, v58, v240
	v_dual_fmac_f32 v62, v68, v240 :: v_dual_fmac_f32 v63, v78, v240
	v_dual_fmac_f32 v72, v88, v240 :: v_dual_fmac_f32 v73, v98, v240
	v_dual_fmac_f32 v82, v108, v240 :: v_dual_fmac_f32 v83, v118, v240
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v92, v48, v248 :: v_dual_fmac_f32 v93, v58, v248
	v_dual_fmac_f32 v102, v68, v248 :: v_dual_fmac_f32 v103, v78, v248
	v_dual_fmac_f32 v112, v88, v248 :: v_dual_fmac_f32 v113, v98, v248
	v_dual_fmac_f32 v122, v108, v248 :: v_dual_fmac_f32 v123, v118, v248
	v_dual_fmac_f32 v52, v49, v241 :: v_dual_fmac_f32 v53, v59, v241
	v_dual_fmac_f32 v62, v69, v241 :: v_dual_fmac_f32 v63, v79, v241
	v_dual_fmac_f32 v72, v89, v241 :: v_dual_fmac_f32 v73, v99, v241
	v_dual_fmac_f32 v82, v109, v241 :: v_dual_fmac_f32 v83, v119, v241
	v_dual_fmac_f32 v92, v49, v249 :: v_dual_fmac_f32 v93, v59, v249
	v_dual_fmac_f32 v102, v69, v249 :: v_dual_fmac_f32 v103, v79, v249
	v_dual_fmac_f32 v112, v89, v249 :: v_dual_fmac_f32 v113, v99, v249
	v_dual_fmac_f32 v122, v109, v249 :: v_dual_fmac_f32 v123, v119, v249
	v_dual_fmac_f32 v52, v50, v242 :: v_dual_fmac_f32 v53, v60, v242
	v_dual_fmac_f32 v62, v70, v242 :: v_dual_fmac_f32 v63, v80, v242
	v_dual_fmac_f32 v72, v90, v242 :: v_dual_fmac_f32 v73, v100, v242
	v_dual_fmac_f32 v82, v110, v242 :: v_dual_fmac_f32 v83, v120, v242
	v_dual_fmac_f32 v92, v50, v250 :: v_dual_fmac_f32 v93, v60, v250
	v_dual_fmac_f32 v102, v70, v250 :: v_dual_fmac_f32 v103, v80, v250
	v_dual_fmac_f32 v112, v90, v250 :: v_dual_fmac_f32 v113, v100, v250
	v_dual_fmac_f32 v122, v110, v250 :: v_dual_fmac_f32 v123, v120, v250
	v_dual_fmac_f32 v52, v51, v243 :: v_dual_fmac_f32 v53, v61, v243
	v_dual_fmac_f32 v62, v71, v243 :: v_dual_fmac_f32 v63, v81, v243
	v_dual_fmac_f32 v72, v91, v243 :: v_dual_fmac_f32 v73, v101, v243
	v_dual_fmac_f32 v82, v111, v243 :: v_dual_fmac_f32 v83, v121, v243
	v_dual_fmac_f32 v92, v51, v251 :: v_dual_fmac_f32 v93, v61, v251
	v_dual_fmac_f32 v102, v71, v251 :: v_dual_fmac_f32 v103, v81, v251
	v_dual_fmac_f32 v112, v91, v251 :: v_dual_fmac_f32 v113, v101, v251
	v_dual_fmac_f32 v122, v111, v251 :: v_dual_fmac_f32 v123, v121, v251
	global_load_b128 v[236:239], v3, s[32:33]
	global_load_b128 v[240:243], v3, s[32:33] offset:16
	global_load_b128 v[244:247], v3, s[34:35]
	global_load_b128 v[248:251], v3, s[34:35] offset:16
	v_dual_add_f32 v172, v52, v172 :: v_dual_add_f32 v173, v173, v53
	v_dual_add_f32 v174, v62, v174 :: v_dual_add_f32 v175, v175, v63
	v_dual_add_f32 v176, v72, v176 :: v_dual_add_f32 v177, v177, v73
	v_dual_add_f32 v178, v82, v178 :: v_dual_add_f32 v179, v179, v83
	v_dual_add_f32 v196, v92, v196 :: v_dual_add_f32 v197, v197, v93
	v_dual_add_f32 v198, v102, v198 :: v_dual_add_f32 v199, v199, v103
	v_dual_add_f32 v200, v112, v200 :: v_dual_add_f32 v201, v201, v113
	v_dual_add_f32 v202, v122, v202 :: v_dual_add_f32 v203, v203, v123
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s0
	s_wait_loadcnt 0x0
	.Lrx_b4_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[220:223], v2, s[28:29]
	global_load_b128 v[224:227], v2, s[28:29] offset:16
	global_load_b128 v[228:231], v2, s[30:31]
	global_load_b128 v[232:235], v2, s[30:31] offset:16
	global_load_b128 v[236:239], v2, s[32:33]
	global_load_b128 v[240:243], v2, s[32:33] offset:16
	global_load_b128 v[244:247], v2, s[34:35]
	global_load_b128 v[248:251], v2, s[34:35] offset:16
	.Lrx_b4_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x16
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v52, v44, v220 :: v_dual_mul_f32 v53, v55, v221
	v_dual_mul_f32 v62, v64, v220 :: v_dual_mul_f32 v63, v75, v221
	v_dual_mul_f32 v72, v84, v220 :: v_dual_mul_f32 v73, v95, v221
	v_dual_mul_f32 v82, v104, v220 :: v_dual_mul_f32 v83, v115, v221
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v92, v44, v228 :: v_dual_mul_f32 v93, v55, v229
	v_dual_mul_f32 v102, v64, v228 :: v_dual_mul_f32 v103, v75, v229
	v_dual_mul_f32 v112, v84, v228 :: v_dual_mul_f32 v113, v95, v229
	v_dual_mul_f32 v122, v104, v228 :: v_dual_mul_f32 v123, v115, v229
	v_dual_fmac_f32 v52, v45, v221 :: v_dual_fmac_f32 v53, v54, v220
	v_dual_fmac_f32 v62, v65, v221 :: v_dual_fmac_f32 v63, v74, v220
	v_dual_fmac_f32 v72, v85, v221 :: v_dual_fmac_f32 v73, v94, v220
	v_dual_fmac_f32 v82, v105, v221 :: v_dual_fmac_f32 v83, v114, v220
	v_dual_fmac_f32 v92, v45, v229 :: v_dual_fmac_f32 v93, v54, v228
	v_dual_fmac_f32 v102, v65, v229 :: v_dual_fmac_f32 v103, v74, v228
	v_dual_fmac_f32 v112, v85, v229 :: v_dual_fmac_f32 v113, v94, v228
	v_dual_fmac_f32 v122, v105, v229 :: v_dual_fmac_f32 v123, v114, v228
	v_dual_fmac_f32 v52, v46, v222 :: v_dual_fmac_f32 v53, v56, v222
	v_dual_fmac_f32 v62, v66, v222 :: v_dual_fmac_f32 v63, v76, v222
	v_dual_fmac_f32 v72, v86, v222 :: v_dual_fmac_f32 v73, v96, v222
	v_dual_fmac_f32 v82, v106, v222 :: v_dual_fmac_f32 v83, v116, v222
	v_dual_fmac_f32 v92, v46, v230 :: v_dual_fmac_f32 v93, v56, v230
	v_dual_fmac_f32 v102, v66, v230 :: v_dual_fmac_f32 v103, v76, v230
	v_dual_fmac_f32 v112, v86, v230 :: v_dual_fmac_f32 v113, v96, v230
	v_dual_fmac_f32 v122, v106, v230 :: v_dual_fmac_f32 v123, v116, v230
	v_dual_fmac_f32 v52, v47, v223 :: v_dual_fmac_f32 v53, v57, v223
	v_dual_fmac_f32 v62, v67, v223 :: v_dual_fmac_f32 v63, v77, v223
	v_dual_fmac_f32 v72, v87, v223 :: v_dual_fmac_f32 v73, v97, v223
	v_dual_fmac_f32 v82, v107, v223 :: v_dual_fmac_f32 v83, v117, v223
	v_dual_fmac_f32 v92, v47, v231 :: v_dual_fmac_f32 v93, v57, v231
	v_dual_fmac_f32 v102, v67, v231 :: v_dual_fmac_f32 v103, v77, v231
	v_dual_fmac_f32 v112, v87, v231 :: v_dual_fmac_f32 v113, v97, v231
	v_dual_fmac_f32 v122, v107, v231 :: v_dual_fmac_f32 v123, v117, v231
	v_dual_fmac_f32 v52, v48, v224 :: v_dual_fmac_f32 v53, v58, v224
	v_dual_fmac_f32 v62, v68, v224 :: v_dual_fmac_f32 v63, v78, v224
	v_dual_fmac_f32 v72, v88, v224 :: v_dual_fmac_f32 v73, v98, v224
	v_dual_fmac_f32 v82, v108, v224 :: v_dual_fmac_f32 v83, v118, v224
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v92, v48, v232 :: v_dual_fmac_f32 v93, v58, v232
	v_dual_fmac_f32 v102, v68, v232 :: v_dual_fmac_f32 v103, v78, v232
	v_dual_fmac_f32 v112, v88, v232 :: v_dual_fmac_f32 v113, v98, v232
	v_dual_fmac_f32 v122, v108, v232 :: v_dual_fmac_f32 v123, v118, v232
	v_dual_fmac_f32 v52, v49, v225 :: v_dual_fmac_f32 v53, v59, v225
	v_dual_fmac_f32 v62, v69, v225 :: v_dual_fmac_f32 v63, v79, v225
	v_dual_fmac_f32 v72, v89, v225 :: v_dual_fmac_f32 v73, v99, v225
	v_dual_fmac_f32 v82, v109, v225 :: v_dual_fmac_f32 v83, v119, v225
	v_dual_fmac_f32 v92, v49, v233 :: v_dual_fmac_f32 v93, v59, v233
	v_dual_fmac_f32 v102, v69, v233 :: v_dual_fmac_f32 v103, v79, v233
	v_dual_fmac_f32 v112, v89, v233 :: v_dual_fmac_f32 v113, v99, v233
	v_dual_fmac_f32 v122, v109, v233 :: v_dual_fmac_f32 v123, v119, v233
	v_dual_fmac_f32 v52, v50, v226 :: v_dual_fmac_f32 v53, v60, v226
	v_dual_fmac_f32 v62, v70, v226 :: v_dual_fmac_f32 v63, v80, v226
	v_dual_fmac_f32 v72, v90, v226 :: v_dual_fmac_f32 v73, v100, v226
	v_dual_fmac_f32 v82, v110, v226 :: v_dual_fmac_f32 v83, v120, v226
	v_dual_fmac_f32 v92, v50, v234 :: v_dual_fmac_f32 v93, v60, v234
	v_dual_fmac_f32 v102, v70, v234 :: v_dual_fmac_f32 v103, v80, v234
	v_dual_fmac_f32 v112, v90, v234 :: v_dual_fmac_f32 v113, v100, v234
	v_dual_fmac_f32 v122, v110, v234 :: v_dual_fmac_f32 v123, v120, v234
	v_dual_fmac_f32 v52, v51, v227 :: v_dual_fmac_f32 v53, v61, v227
	v_dual_fmac_f32 v62, v71, v227 :: v_dual_fmac_f32 v63, v81, v227
	v_dual_fmac_f32 v72, v91, v227 :: v_dual_fmac_f32 v73, v101, v227
	v_dual_fmac_f32 v82, v111, v227 :: v_dual_fmac_f32 v83, v121, v227
	v_dual_fmac_f32 v92, v51, v235 :: v_dual_fmac_f32 v93, v61, v235
	v_dual_fmac_f32 v102, v71, v235 :: v_dual_fmac_f32 v103, v81, v235
	v_dual_fmac_f32 v112, v91, v235 :: v_dual_fmac_f32 v113, v101, v235
	v_dual_fmac_f32 v122, v111, v235 :: v_dual_fmac_f32 v123, v121, v235
	global_load_b128 v[220:223], v3, s[28:29]
	global_load_b128 v[224:227], v3, s[28:29] offset:16
	global_load_b128 v[228:231], v3, s[30:31]
	global_load_b128 v[232:235], v3, s[30:31] offset:16
	v_dual_add_f32 v132, v52, v132 :: v_dual_add_f32 v133, v133, v53
	v_dual_add_f32 v134, v62, v134 :: v_dual_add_f32 v135, v135, v63
	v_dual_add_f32 v136, v72, v136 :: v_dual_add_f32 v137, v137, v73
	v_dual_add_f32 v138, v82, v138 :: v_dual_add_f32 v139, v139, v83
	v_dual_add_f32 v156, v92, v156 :: v_dual_add_f32 v157, v157, v93
	v_dual_add_f32 v158, v102, v158 :: v_dual_add_f32 v159, v159, v103
	v_dual_add_f32 v160, v112, v160 :: v_dual_add_f32 v161, v161, v113
	v_dual_add_f32 v162, v122, v162 :: v_dual_add_f32 v163, v163, v123
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v52, v44, v236 :: v_dual_mul_f32 v53, v55, v237
	v_dual_mul_f32 v62, v64, v236 :: v_dual_mul_f32 v63, v75, v237
	v_dual_mul_f32 v72, v84, v236 :: v_dual_mul_f32 v73, v95, v237
	v_dual_mul_f32 v82, v104, v236 :: v_dual_mul_f32 v83, v115, v237
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v92, v44, v244 :: v_dual_mul_f32 v93, v55, v245
	v_dual_mul_f32 v102, v64, v244 :: v_dual_mul_f32 v103, v75, v245
	v_dual_mul_f32 v112, v84, v244 :: v_dual_mul_f32 v113, v95, v245
	v_dual_mul_f32 v122, v104, v244 :: v_dual_mul_f32 v123, v115, v245
	v_dual_fmac_f32 v52, v45, v237 :: v_dual_fmac_f32 v53, v54, v236
	v_dual_fmac_f32 v62, v65, v237 :: v_dual_fmac_f32 v63, v74, v236
	v_dual_fmac_f32 v72, v85, v237 :: v_dual_fmac_f32 v73, v94, v236
	v_dual_fmac_f32 v82, v105, v237 :: v_dual_fmac_f32 v83, v114, v236
	v_dual_fmac_f32 v92, v45, v245 :: v_dual_fmac_f32 v93, v54, v244
	v_dual_fmac_f32 v102, v65, v245 :: v_dual_fmac_f32 v103, v74, v244
	v_dual_fmac_f32 v112, v85, v245 :: v_dual_fmac_f32 v113, v94, v244
	v_dual_fmac_f32 v122, v105, v245 :: v_dual_fmac_f32 v123, v114, v244
	v_dual_fmac_f32 v52, v46, v238 :: v_dual_fmac_f32 v53, v56, v238
	v_dual_fmac_f32 v62, v66, v238 :: v_dual_fmac_f32 v63, v76, v238
	v_dual_fmac_f32 v72, v86, v238 :: v_dual_fmac_f32 v73, v96, v238
	v_dual_fmac_f32 v82, v106, v238 :: v_dual_fmac_f32 v83, v116, v238
	v_dual_fmac_f32 v92, v46, v246 :: v_dual_fmac_f32 v93, v56, v246
	v_dual_fmac_f32 v102, v66, v246 :: v_dual_fmac_f32 v103, v76, v246
	v_dual_fmac_f32 v112, v86, v246 :: v_dual_fmac_f32 v113, v96, v246
	v_dual_fmac_f32 v122, v106, v246 :: v_dual_fmac_f32 v123, v116, v246
	v_dual_fmac_f32 v52, v47, v239 :: v_dual_fmac_f32 v53, v57, v239
	v_dual_fmac_f32 v62, v67, v239 :: v_dual_fmac_f32 v63, v77, v239
	v_dual_fmac_f32 v72, v87, v239 :: v_dual_fmac_f32 v73, v97, v239
	v_dual_fmac_f32 v82, v107, v239 :: v_dual_fmac_f32 v83, v117, v239
	v_dual_fmac_f32 v92, v47, v247 :: v_dual_fmac_f32 v93, v57, v247
	v_dual_fmac_f32 v102, v67, v247 :: v_dual_fmac_f32 v103, v77, v247
	v_dual_fmac_f32 v112, v87, v247 :: v_dual_fmac_f32 v113, v97, v247
	v_dual_fmac_f32 v122, v107, v247 :: v_dual_fmac_f32 v123, v117, v247
	v_dual_fmac_f32 v52, v48, v240 :: v_dual_fmac_f32 v53, v58, v240
	v_dual_fmac_f32 v62, v68, v240 :: v_dual_fmac_f32 v63, v78, v240
	v_dual_fmac_f32 v72, v88, v240 :: v_dual_fmac_f32 v73, v98, v240
	v_dual_fmac_f32 v82, v108, v240 :: v_dual_fmac_f32 v83, v118, v240
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v92, v48, v248 :: v_dual_fmac_f32 v93, v58, v248
	v_dual_fmac_f32 v102, v68, v248 :: v_dual_fmac_f32 v103, v78, v248
	v_dual_fmac_f32 v112, v88, v248 :: v_dual_fmac_f32 v113, v98, v248
	v_dual_fmac_f32 v122, v108, v248 :: v_dual_fmac_f32 v123, v118, v248
	v_dual_fmac_f32 v52, v49, v241 :: v_dual_fmac_f32 v53, v59, v241
	v_dual_fmac_f32 v62, v69, v241 :: v_dual_fmac_f32 v63, v79, v241
	v_dual_fmac_f32 v72, v89, v241 :: v_dual_fmac_f32 v73, v99, v241
	v_dual_fmac_f32 v82, v109, v241 :: v_dual_fmac_f32 v83, v119, v241
	v_dual_fmac_f32 v92, v49, v249 :: v_dual_fmac_f32 v93, v59, v249
	v_dual_fmac_f32 v102, v69, v249 :: v_dual_fmac_f32 v103, v79, v249
	v_dual_fmac_f32 v112, v89, v249 :: v_dual_fmac_f32 v113, v99, v249
	v_dual_fmac_f32 v122, v109, v249 :: v_dual_fmac_f32 v123, v119, v249
	v_dual_fmac_f32 v52, v50, v242 :: v_dual_fmac_f32 v53, v60, v242
	v_dual_fmac_f32 v62, v70, v242 :: v_dual_fmac_f32 v63, v80, v242
	v_dual_fmac_f32 v72, v90, v242 :: v_dual_fmac_f32 v73, v100, v242
	v_dual_fmac_f32 v82, v110, v242 :: v_dual_fmac_f32 v83, v120, v242
	v_dual_fmac_f32 v92, v50, v250 :: v_dual_fmac_f32 v93, v60, v250
	v_dual_fmac_f32 v102, v70, v250 :: v_dual_fmac_f32 v103, v80, v250
	v_dual_fmac_f32 v112, v90, v250 :: v_dual_fmac_f32 v113, v100, v250
	v_dual_fmac_f32 v122, v110, v250 :: v_dual_fmac_f32 v123, v120, v250
	v_dual_fmac_f32 v52, v51, v243 :: v_dual_fmac_f32 v53, v61, v243
	v_dual_fmac_f32 v62, v71, v243 :: v_dual_fmac_f32 v63, v81, v243
	v_dual_fmac_f32 v72, v91, v243 :: v_dual_fmac_f32 v73, v101, v243
	v_dual_fmac_f32 v82, v111, v243 :: v_dual_fmac_f32 v83, v121, v243
	v_dual_fmac_f32 v92, v51, v251 :: v_dual_fmac_f32 v93, v61, v251
	v_dual_fmac_f32 v102, v71, v251 :: v_dual_fmac_f32 v103, v81, v251
	v_dual_fmac_f32 v112, v91, v251 :: v_dual_fmac_f32 v113, v101, v251
	v_dual_fmac_f32 v122, v111, v251 :: v_dual_fmac_f32 v123, v121, v251
	global_load_b128 v[236:239], v3, s[32:33]
	global_load_b128 v[240:243], v3, s[32:33] offset:16
	global_load_b128 v[244:247], v3, s[34:35]
	global_load_b128 v[248:251], v3, s[34:35] offset:16
	v_dual_add_f32 v180, v52, v180 :: v_dual_add_f32 v181, v181, v53
	v_dual_add_f32 v182, v62, v182 :: v_dual_add_f32 v183, v183, v63
	v_dual_add_f32 v184, v72, v184 :: v_dual_add_f32 v185, v185, v73
	v_dual_add_f32 v186, v82, v186 :: v_dual_add_f32 v187, v187, v83
	v_dual_add_f32 v204, v92, v204 :: v_dual_add_f32 v205, v205, v93
	v_dual_add_f32 v206, v102, v206 :: v_dual_add_f32 v207, v207, v103
	v_dual_add_f32 v208, v112, v208 :: v_dual_add_f32 v209, v209, v113
	v_dual_add_f32 v210, v122, v210 :: v_dual_add_f32 v211, v211, v123
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s1
	s_wait_loadcnt 0x0
	.Lrx_b4_s1_done:
	v_dual_add_f32 v124, v124, v132 :: v_dual_add_f32 v125, v125, v133
	v_dual_add_f32 v126, v126, v134 :: v_dual_add_f32 v127, v127, v135
	v_dual_add_f32 v128, v128, v136 :: v_dual_add_f32 v129, v129, v137
	v_dual_add_f32 v130, v130, v138 :: v_dual_add_f32 v131, v131, v139
	v_dual_add_f32 v148, v148, v156 :: v_dual_add_f32 v149, v149, v157
	v_dual_add_f32 v150, v150, v158 :: v_dual_add_f32 v151, v151, v159
	v_dual_add_f32 v152, v152, v160 :: v_dual_add_f32 v153, v153, v161
	v_dual_add_f32 v154, v154, v162 :: v_dual_add_f32 v155, v155, v163
	v_dual_add_f32 v172, v172, v180 :: v_dual_add_f32 v173, v173, v181
	v_dual_add_f32 v174, v174, v182 :: v_dual_add_f32 v175, v175, v183
	v_dual_add_f32 v176, v176, v184 :: v_dual_add_f32 v177, v177, v185
	v_dual_add_f32 v178, v178, v186 :: v_dual_add_f32 v179, v179, v187
	v_dual_add_f32 v196, v196, v204 :: v_dual_add_f32 v197, v197, v205
	v_dual_add_f32 v198, v198, v206 :: v_dual_add_f32 v199, v199, v207
	v_dual_add_f32 v200, v200, v208 :: v_dual_add_f32 v201, v201, v209
	v_dual_add_f32 v202, v202, v210 :: v_dual_add_f32 v203, v203, v211
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[220:223], v2, s[28:29]
	global_load_b128 v[224:227], v2, s[28:29] offset:16
	global_load_b128 v[228:231], v2, s[30:31]
	global_load_b128 v[232:235], v2, s[30:31] offset:16
	global_load_b128 v[236:239], v2, s[32:33]
	global_load_b128 v[240:243], v2, s[32:33] offset:16
	global_load_b128 v[244:247], v2, s[34:35]
	global_load_b128 v[248:251], v2, s[34:35] offset:16
	.Lrx_b4_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x16
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v52, v44, v220 :: v_dual_mul_f32 v53, v55, v221
	v_dual_mul_f32 v62, v64, v220 :: v_dual_mul_f32 v63, v75, v221
	v_dual_mul_f32 v72, v84, v220 :: v_dual_mul_f32 v73, v95, v221
	v_dual_mul_f32 v82, v104, v220 :: v_dual_mul_f32 v83, v115, v221
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v92, v44, v228 :: v_dual_mul_f32 v93, v55, v229
	v_dual_mul_f32 v102, v64, v228 :: v_dual_mul_f32 v103, v75, v229
	v_dual_mul_f32 v112, v84, v228 :: v_dual_mul_f32 v113, v95, v229
	v_dual_mul_f32 v122, v104, v228 :: v_dual_mul_f32 v123, v115, v229
	v_dual_fmac_f32 v52, v45, v221 :: v_dual_fmac_f32 v53, v54, v220
	v_dual_fmac_f32 v62, v65, v221 :: v_dual_fmac_f32 v63, v74, v220
	v_dual_fmac_f32 v72, v85, v221 :: v_dual_fmac_f32 v73, v94, v220
	v_dual_fmac_f32 v82, v105, v221 :: v_dual_fmac_f32 v83, v114, v220
	v_dual_fmac_f32 v92, v45, v229 :: v_dual_fmac_f32 v93, v54, v228
	v_dual_fmac_f32 v102, v65, v229 :: v_dual_fmac_f32 v103, v74, v228
	v_dual_fmac_f32 v112, v85, v229 :: v_dual_fmac_f32 v113, v94, v228
	v_dual_fmac_f32 v122, v105, v229 :: v_dual_fmac_f32 v123, v114, v228
	v_dual_fmac_f32 v52, v46, v222 :: v_dual_fmac_f32 v53, v56, v222
	v_dual_fmac_f32 v62, v66, v222 :: v_dual_fmac_f32 v63, v76, v222
	v_dual_fmac_f32 v72, v86, v222 :: v_dual_fmac_f32 v73, v96, v222
	v_dual_fmac_f32 v82, v106, v222 :: v_dual_fmac_f32 v83, v116, v222
	v_dual_fmac_f32 v92, v46, v230 :: v_dual_fmac_f32 v93, v56, v230
	v_dual_fmac_f32 v102, v66, v230 :: v_dual_fmac_f32 v103, v76, v230
	v_dual_fmac_f32 v112, v86, v230 :: v_dual_fmac_f32 v113, v96, v230
	v_dual_fmac_f32 v122, v106, v230 :: v_dual_fmac_f32 v123, v116, v230
	v_dual_fmac_f32 v52, v47, v223 :: v_dual_fmac_f32 v53, v57, v223
	v_dual_fmac_f32 v62, v67, v223 :: v_dual_fmac_f32 v63, v77, v223
	v_dual_fmac_f32 v72, v87, v223 :: v_dual_fmac_f32 v73, v97, v223
	v_dual_fmac_f32 v82, v107, v223 :: v_dual_fmac_f32 v83, v117, v223
	v_dual_fmac_f32 v92, v47, v231 :: v_dual_fmac_f32 v93, v57, v231
	v_dual_fmac_f32 v102, v67, v231 :: v_dual_fmac_f32 v103, v77, v231
	v_dual_fmac_f32 v112, v87, v231 :: v_dual_fmac_f32 v113, v97, v231
	v_dual_fmac_f32 v122, v107, v231 :: v_dual_fmac_f32 v123, v117, v231
	v_dual_fmac_f32 v52, v48, v224 :: v_dual_fmac_f32 v53, v58, v224
	v_dual_fmac_f32 v62, v68, v224 :: v_dual_fmac_f32 v63, v78, v224
	v_dual_fmac_f32 v72, v88, v224 :: v_dual_fmac_f32 v73, v98, v224
	v_dual_fmac_f32 v82, v108, v224 :: v_dual_fmac_f32 v83, v118, v224
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v92, v48, v232 :: v_dual_fmac_f32 v93, v58, v232
	v_dual_fmac_f32 v102, v68, v232 :: v_dual_fmac_f32 v103, v78, v232
	v_dual_fmac_f32 v112, v88, v232 :: v_dual_fmac_f32 v113, v98, v232
	v_dual_fmac_f32 v122, v108, v232 :: v_dual_fmac_f32 v123, v118, v232
	v_dual_fmac_f32 v52, v49, v225 :: v_dual_fmac_f32 v53, v59, v225
	v_dual_fmac_f32 v62, v69, v225 :: v_dual_fmac_f32 v63, v79, v225
	v_dual_fmac_f32 v72, v89, v225 :: v_dual_fmac_f32 v73, v99, v225
	v_dual_fmac_f32 v82, v109, v225 :: v_dual_fmac_f32 v83, v119, v225
	v_dual_fmac_f32 v92, v49, v233 :: v_dual_fmac_f32 v93, v59, v233
	v_dual_fmac_f32 v102, v69, v233 :: v_dual_fmac_f32 v103, v79, v233
	v_dual_fmac_f32 v112, v89, v233 :: v_dual_fmac_f32 v113, v99, v233
	v_dual_fmac_f32 v122, v109, v233 :: v_dual_fmac_f32 v123, v119, v233
	v_dual_fmac_f32 v52, v50, v226 :: v_dual_fmac_f32 v53, v60, v226
	v_dual_fmac_f32 v62, v70, v226 :: v_dual_fmac_f32 v63, v80, v226
	v_dual_fmac_f32 v72, v90, v226 :: v_dual_fmac_f32 v73, v100, v226
	v_dual_fmac_f32 v82, v110, v226 :: v_dual_fmac_f32 v83, v120, v226
	v_dual_fmac_f32 v92, v50, v234 :: v_dual_fmac_f32 v93, v60, v234
	v_dual_fmac_f32 v102, v70, v234 :: v_dual_fmac_f32 v103, v80, v234
	v_dual_fmac_f32 v112, v90, v234 :: v_dual_fmac_f32 v113, v100, v234
	v_dual_fmac_f32 v122, v110, v234 :: v_dual_fmac_f32 v123, v120, v234
	v_dual_fmac_f32 v52, v51, v227 :: v_dual_fmac_f32 v53, v61, v227
	v_dual_fmac_f32 v62, v71, v227 :: v_dual_fmac_f32 v63, v81, v227
	v_dual_fmac_f32 v72, v91, v227 :: v_dual_fmac_f32 v73, v101, v227
	v_dual_fmac_f32 v82, v111, v227 :: v_dual_fmac_f32 v83, v121, v227
	v_dual_fmac_f32 v92, v51, v235 :: v_dual_fmac_f32 v93, v61, v235
	v_dual_fmac_f32 v102, v71, v235 :: v_dual_fmac_f32 v103, v81, v235
	v_dual_fmac_f32 v112, v91, v235 :: v_dual_fmac_f32 v113, v101, v235
	v_dual_fmac_f32 v122, v111, v235 :: v_dual_fmac_f32 v123, v121, v235
	global_load_b128 v[220:223], v3, s[28:29]
	global_load_b128 v[224:227], v3, s[28:29] offset:16
	global_load_b128 v[228:231], v3, s[30:31]
	global_load_b128 v[232:235], v3, s[30:31] offset:16
	v_dual_add_f32 v132, v52, v132 :: v_dual_add_f32 v133, v133, v53
	v_dual_add_f32 v134, v62, v134 :: v_dual_add_f32 v135, v135, v63
	v_dual_add_f32 v136, v72, v136 :: v_dual_add_f32 v137, v137, v73
	v_dual_add_f32 v138, v82, v138 :: v_dual_add_f32 v139, v139, v83
	v_dual_add_f32 v156, v92, v156 :: v_dual_add_f32 v157, v157, v93
	v_dual_add_f32 v158, v102, v158 :: v_dual_add_f32 v159, v159, v103
	v_dual_add_f32 v160, v112, v160 :: v_dual_add_f32 v161, v161, v113
	v_dual_add_f32 v162, v122, v162 :: v_dual_add_f32 v163, v163, v123
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v52, v44, v236 :: v_dual_mul_f32 v53, v55, v237
	v_dual_mul_f32 v62, v64, v236 :: v_dual_mul_f32 v63, v75, v237
	v_dual_mul_f32 v72, v84, v236 :: v_dual_mul_f32 v73, v95, v237
	v_dual_mul_f32 v82, v104, v236 :: v_dual_mul_f32 v83, v115, v237
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v92, v44, v244 :: v_dual_mul_f32 v93, v55, v245
	v_dual_mul_f32 v102, v64, v244 :: v_dual_mul_f32 v103, v75, v245
	v_dual_mul_f32 v112, v84, v244 :: v_dual_mul_f32 v113, v95, v245
	v_dual_mul_f32 v122, v104, v244 :: v_dual_mul_f32 v123, v115, v245
	v_dual_fmac_f32 v52, v45, v237 :: v_dual_fmac_f32 v53, v54, v236
	v_dual_fmac_f32 v62, v65, v237 :: v_dual_fmac_f32 v63, v74, v236
	v_dual_fmac_f32 v72, v85, v237 :: v_dual_fmac_f32 v73, v94, v236
	v_dual_fmac_f32 v82, v105, v237 :: v_dual_fmac_f32 v83, v114, v236
	v_dual_fmac_f32 v92, v45, v245 :: v_dual_fmac_f32 v93, v54, v244
	v_dual_fmac_f32 v102, v65, v245 :: v_dual_fmac_f32 v103, v74, v244
	v_dual_fmac_f32 v112, v85, v245 :: v_dual_fmac_f32 v113, v94, v244
	v_dual_fmac_f32 v122, v105, v245 :: v_dual_fmac_f32 v123, v114, v244
	v_dual_fmac_f32 v52, v46, v238 :: v_dual_fmac_f32 v53, v56, v238
	v_dual_fmac_f32 v62, v66, v238 :: v_dual_fmac_f32 v63, v76, v238
	v_dual_fmac_f32 v72, v86, v238 :: v_dual_fmac_f32 v73, v96, v238
	v_dual_fmac_f32 v82, v106, v238 :: v_dual_fmac_f32 v83, v116, v238
	v_dual_fmac_f32 v92, v46, v246 :: v_dual_fmac_f32 v93, v56, v246
	v_dual_fmac_f32 v102, v66, v246 :: v_dual_fmac_f32 v103, v76, v246
	v_dual_fmac_f32 v112, v86, v246 :: v_dual_fmac_f32 v113, v96, v246
	v_dual_fmac_f32 v122, v106, v246 :: v_dual_fmac_f32 v123, v116, v246
	v_dual_fmac_f32 v52, v47, v239 :: v_dual_fmac_f32 v53, v57, v239
	v_dual_fmac_f32 v62, v67, v239 :: v_dual_fmac_f32 v63, v77, v239
	v_dual_fmac_f32 v72, v87, v239 :: v_dual_fmac_f32 v73, v97, v239
	v_dual_fmac_f32 v82, v107, v239 :: v_dual_fmac_f32 v83, v117, v239
	v_dual_fmac_f32 v92, v47, v247 :: v_dual_fmac_f32 v93, v57, v247
	v_dual_fmac_f32 v102, v67, v247 :: v_dual_fmac_f32 v103, v77, v247
	v_dual_fmac_f32 v112, v87, v247 :: v_dual_fmac_f32 v113, v97, v247
	v_dual_fmac_f32 v122, v107, v247 :: v_dual_fmac_f32 v123, v117, v247
	v_dual_fmac_f32 v52, v48, v240 :: v_dual_fmac_f32 v53, v58, v240
	v_dual_fmac_f32 v62, v68, v240 :: v_dual_fmac_f32 v63, v78, v240
	v_dual_fmac_f32 v72, v88, v240 :: v_dual_fmac_f32 v73, v98, v240
	v_dual_fmac_f32 v82, v108, v240 :: v_dual_fmac_f32 v83, v118, v240
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v92, v48, v248 :: v_dual_fmac_f32 v93, v58, v248
	v_dual_fmac_f32 v102, v68, v248 :: v_dual_fmac_f32 v103, v78, v248
	v_dual_fmac_f32 v112, v88, v248 :: v_dual_fmac_f32 v113, v98, v248
	v_dual_fmac_f32 v122, v108, v248 :: v_dual_fmac_f32 v123, v118, v248
	v_dual_fmac_f32 v52, v49, v241 :: v_dual_fmac_f32 v53, v59, v241
	v_dual_fmac_f32 v62, v69, v241 :: v_dual_fmac_f32 v63, v79, v241
	v_dual_fmac_f32 v72, v89, v241 :: v_dual_fmac_f32 v73, v99, v241
	v_dual_fmac_f32 v82, v109, v241 :: v_dual_fmac_f32 v83, v119, v241
	v_dual_fmac_f32 v92, v49, v249 :: v_dual_fmac_f32 v93, v59, v249
	v_dual_fmac_f32 v102, v69, v249 :: v_dual_fmac_f32 v103, v79, v249
	v_dual_fmac_f32 v112, v89, v249 :: v_dual_fmac_f32 v113, v99, v249
	v_dual_fmac_f32 v122, v109, v249 :: v_dual_fmac_f32 v123, v119, v249
	v_dual_fmac_f32 v52, v50, v242 :: v_dual_fmac_f32 v53, v60, v242
	v_dual_fmac_f32 v62, v70, v242 :: v_dual_fmac_f32 v63, v80, v242
	v_dual_fmac_f32 v72, v90, v242 :: v_dual_fmac_f32 v73, v100, v242
	v_dual_fmac_f32 v82, v110, v242 :: v_dual_fmac_f32 v83, v120, v242
	v_dual_fmac_f32 v92, v50, v250 :: v_dual_fmac_f32 v93, v60, v250
	v_dual_fmac_f32 v102, v70, v250 :: v_dual_fmac_f32 v103, v80, v250
	v_dual_fmac_f32 v112, v90, v250 :: v_dual_fmac_f32 v113, v100, v250
	v_dual_fmac_f32 v122, v110, v250 :: v_dual_fmac_f32 v123, v120, v250
	v_dual_fmac_f32 v52, v51, v243 :: v_dual_fmac_f32 v53, v61, v243
	v_dual_fmac_f32 v62, v71, v243 :: v_dual_fmac_f32 v63, v81, v243
	v_dual_fmac_f32 v72, v91, v243 :: v_dual_fmac_f32 v73, v101, v243
	v_dual_fmac_f32 v82, v111, v243 :: v_dual_fmac_f32 v83, v121, v243
	v_dual_fmac_f32 v92, v51, v251 :: v_dual_fmac_f32 v93, v61, v251
	v_dual_fmac_f32 v102, v71, v251 :: v_dual_fmac_f32 v103, v81, v251
	v_dual_fmac_f32 v112, v91, v251 :: v_dual_fmac_f32 v113, v101, v251
	v_dual_fmac_f32 v122, v111, v251 :: v_dual_fmac_f32 v123, v121, v251
	global_load_b128 v[236:239], v3, s[32:33]
	global_load_b128 v[240:243], v3, s[32:33] offset:16
	global_load_b128 v[244:247], v3, s[34:35]
	global_load_b128 v[248:251], v3, s[34:35] offset:16
	v_dual_add_f32 v180, v52, v180 :: v_dual_add_f32 v181, v181, v53
	v_dual_add_f32 v182, v62, v182 :: v_dual_add_f32 v183, v183, v63
	v_dual_add_f32 v184, v72, v184 :: v_dual_add_f32 v185, v185, v73
	v_dual_add_f32 v186, v82, v186 :: v_dual_add_f32 v187, v187, v83
	v_dual_add_f32 v204, v92, v204 :: v_dual_add_f32 v205, v205, v93
	v_dual_add_f32 v206, v102, v206 :: v_dual_add_f32 v207, v207, v103
	v_dual_add_f32 v208, v112, v208 :: v_dual_add_f32 v209, v209, v113
	v_dual_add_f32 v210, v122, v210 :: v_dual_add_f32 v211, v211, v123
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s2
	s_wait_loadcnt 0x0
	.Lrx_b4_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[220:223], v2, s[28:29]
	global_load_b128 v[224:227], v2, s[28:29] offset:16
	global_load_b128 v[228:231], v2, s[30:31]
	global_load_b128 v[232:235], v2, s[30:31] offset:16
	global_load_b128 v[236:239], v2, s[32:33]
	global_load_b128 v[240:243], v2, s[32:33] offset:16
	global_load_b128 v[244:247], v2, s[34:35]
	global_load_b128 v[248:251], v2, s[34:35] offset:16
	.Lrx_b4_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x16
	v_bfe_u32 v44, v36, 0, 4
	v_bfe_u32 v45, v36, 4, 4
	v_bfe_u32 v46, v36, 8, 4
	v_bfe_u32 v47, v36, 12, 4
	v_bfe_u32 v48, v36, 16, 4
	v_bfe_u32 v49, v36, 20, 4
	v_bfe_u32 v50, v36, 24, 4
	v_bfe_u32 v51, v36, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v20, v48, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v20, v49, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v20, v50, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v20, v51, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v54, v37, 0, 4
	v_bfe_u32 v55, v37, 4, 4
	v_bfe_u32 v56, v37, 8, 4
	v_bfe_u32 v57, v37, 12, 4
	v_bfe_u32 v58, v37, 16, 4
	v_bfe_u32 v59, v37, 20, 4
	v_bfe_u32 v60, v37, 24, 4
	v_bfe_u32 v61, v37, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v22, v54, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v22, v55, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v22, v56, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v22, v57, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v22, v58, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v22, v59, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v64, v38, 0, 4
	v_bfe_u32 v65, v38, 4, 4
	v_bfe_u32 v66, v38, 8, 4
	v_bfe_u32 v67, v38, 12, 4
	v_bfe_u32 v68, v38, 16, 4
	v_bfe_u32 v69, v38, 20, 4
	v_bfe_u32 v70, v38, 24, 4
	v_bfe_u32 v71, v38, 28, 4
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_fma_mix_f32 v64, v24, v64, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v24, v65, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v24, v66, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v24, v67, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v24, v68, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v24, v69, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v24, v70, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v24, v71, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v74, v39, 0, 4
	v_bfe_u32 v75, v39, 4, 4
	v_bfe_u32 v76, v39, 8, 4
	v_bfe_u32 v77, v39, 12, 4
	v_bfe_u32 v78, v39, 16, 4
	v_bfe_u32 v79, v39, 20, 4
	v_bfe_u32 v80, v39, 24, 4
	v_bfe_u32 v81, v39, 28, 4
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_fma_mix_f32 v74, v26, v74, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v26, v75, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v26, v76, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v26, v77, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v26, v78, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v26, v79, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v26, v80, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v26, v81, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v28, v29, v28, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v84, v40, 0, 4
	v_bfe_u32 v85, v40, 4, 4
	v_bfe_u32 v86, v40, 8, 4
	v_bfe_u32 v87, v40, 12, 4
	v_bfe_u32 v88, v40, 16, 4
	v_bfe_u32 v89, v40, 20, 4
	v_bfe_u32 v90, v40, 24, 4
	v_bfe_u32 v91, v40, 28, 4
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_fma_mix_f32 v84, v28, v84, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v85, v28, v85, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v86, v28, v86, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v28, v87, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v28, v88, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v28, v89, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v28, v90, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v28, v91, v28 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v30, v31, v30, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v94, v41, 0, 4
	v_bfe_u32 v95, v41, 4, 4
	v_bfe_u32 v96, v41, 8, 4
	v_bfe_u32 v97, v41, 12, 4
	v_bfe_u32 v98, v41, 16, 4
	v_bfe_u32 v99, v41, 20, 4
	v_bfe_u32 v100, v41, 24, 4
	v_bfe_u32 v101, v41, 28, 4
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_fma_mix_f32 v94, v30, v94, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v95, v30, v95, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v96, v30, v96, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v97, v30, v97, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v98, v30, v98, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v99, v30, v99, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v100, v30, v100, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v101, v30, v101, v30 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v32, v33, v32, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v104, v42, 0, 4
	v_bfe_u32 v105, v42, 4, 4
	v_bfe_u32 v106, v42, 8, 4
	v_bfe_u32 v107, v42, 12, 4
	v_bfe_u32 v108, v42, 16, 4
	v_bfe_u32 v109, v42, 20, 4
	v_bfe_u32 v110, v42, 24, 4
	v_bfe_u32 v111, v42, 28, 4
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_fma_mix_f32 v104, v32, v104, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v105, v32, v105, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v106, v32, v106, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v107, v32, v107, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v108, v32, v108, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v109, v32, v109, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v110, v32, v110, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v111, v32, v111, v32 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v34, v35, v34, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v114, v43, 0, 4
	v_bfe_u32 v115, v43, 4, 4
	v_bfe_u32 v116, v43, 8, 4
	v_bfe_u32 v117, v43, 12, 4
	v_bfe_u32 v118, v43, 16, 4
	v_bfe_u32 v119, v43, 20, 4
	v_bfe_u32 v120, v43, 24, 4
	v_bfe_u32 v121, v43, 28, 4
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f32_ubyte0_e32 v117, v117
	v_cvt_f32_ubyte0_e32 v118, v118
	v_cvt_f32_ubyte0_e32 v119, v119
	v_cvt_f32_ubyte0_e32 v120, v120
	v_cvt_f32_ubyte0_e32 v121, v121
	v_fma_mix_f32 v114, v34, v114, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v115, v34, v115, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v116, v34, v116, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v117, v34, v117, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v118, v34, v118, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v119, v34, v119, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v120, v34, v120, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v121, v34, v121, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v36, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v37, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v38, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v39, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[28:29], v16, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v40, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[30:31], v17, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v41, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[32:33], v18, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v42, v10, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[34:35], v19, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v43, v11, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v52, v44, v220 :: v_dual_mul_f32 v53, v55, v221
	v_dual_mul_f32 v62, v64, v220 :: v_dual_mul_f32 v63, v75, v221
	v_dual_mul_f32 v72, v84, v220 :: v_dual_mul_f32 v73, v95, v221
	v_dual_mul_f32 v82, v104, v220 :: v_dual_mul_f32 v83, v115, v221
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v92, v44, v228 :: v_dual_mul_f32 v93, v55, v229
	v_dual_mul_f32 v102, v64, v228 :: v_dual_mul_f32 v103, v75, v229
	v_dual_mul_f32 v112, v84, v228 :: v_dual_mul_f32 v113, v95, v229
	v_dual_mul_f32 v122, v104, v228 :: v_dual_mul_f32 v123, v115, v229
	v_dual_fmac_f32 v52, v45, v221 :: v_dual_fmac_f32 v53, v54, v220
	v_dual_fmac_f32 v62, v65, v221 :: v_dual_fmac_f32 v63, v74, v220
	v_dual_fmac_f32 v72, v85, v221 :: v_dual_fmac_f32 v73, v94, v220
	v_dual_fmac_f32 v82, v105, v221 :: v_dual_fmac_f32 v83, v114, v220
	v_dual_fmac_f32 v92, v45, v229 :: v_dual_fmac_f32 v93, v54, v228
	v_dual_fmac_f32 v102, v65, v229 :: v_dual_fmac_f32 v103, v74, v228
	v_dual_fmac_f32 v112, v85, v229 :: v_dual_fmac_f32 v113, v94, v228
	v_dual_fmac_f32 v122, v105, v229 :: v_dual_fmac_f32 v123, v114, v228
	v_dual_fmac_f32 v52, v46, v222 :: v_dual_fmac_f32 v53, v56, v222
	v_dual_fmac_f32 v62, v66, v222 :: v_dual_fmac_f32 v63, v76, v222
	v_dual_fmac_f32 v72, v86, v222 :: v_dual_fmac_f32 v73, v96, v222
	v_dual_fmac_f32 v82, v106, v222 :: v_dual_fmac_f32 v83, v116, v222
	v_dual_fmac_f32 v92, v46, v230 :: v_dual_fmac_f32 v93, v56, v230
	v_dual_fmac_f32 v102, v66, v230 :: v_dual_fmac_f32 v103, v76, v230
	v_dual_fmac_f32 v112, v86, v230 :: v_dual_fmac_f32 v113, v96, v230
	v_dual_fmac_f32 v122, v106, v230 :: v_dual_fmac_f32 v123, v116, v230
	v_dual_fmac_f32 v52, v47, v223 :: v_dual_fmac_f32 v53, v57, v223
	v_dual_fmac_f32 v62, v67, v223 :: v_dual_fmac_f32 v63, v77, v223
	v_dual_fmac_f32 v72, v87, v223 :: v_dual_fmac_f32 v73, v97, v223
	v_dual_fmac_f32 v82, v107, v223 :: v_dual_fmac_f32 v83, v117, v223
	v_dual_fmac_f32 v92, v47, v231 :: v_dual_fmac_f32 v93, v57, v231
	v_dual_fmac_f32 v102, v67, v231 :: v_dual_fmac_f32 v103, v77, v231
	v_dual_fmac_f32 v112, v87, v231 :: v_dual_fmac_f32 v113, v97, v231
	v_dual_fmac_f32 v122, v107, v231 :: v_dual_fmac_f32 v123, v117, v231
	v_dual_fmac_f32 v52, v48, v224 :: v_dual_fmac_f32 v53, v58, v224
	v_dual_fmac_f32 v62, v68, v224 :: v_dual_fmac_f32 v63, v78, v224
	v_dual_fmac_f32 v72, v88, v224 :: v_dual_fmac_f32 v73, v98, v224
	v_dual_fmac_f32 v82, v108, v224 :: v_dual_fmac_f32 v83, v118, v224
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v92, v48, v232 :: v_dual_fmac_f32 v93, v58, v232
	v_dual_fmac_f32 v102, v68, v232 :: v_dual_fmac_f32 v103, v78, v232
	v_dual_fmac_f32 v112, v88, v232 :: v_dual_fmac_f32 v113, v98, v232
	v_dual_fmac_f32 v122, v108, v232 :: v_dual_fmac_f32 v123, v118, v232
	v_dual_fmac_f32 v52, v49, v225 :: v_dual_fmac_f32 v53, v59, v225
	v_dual_fmac_f32 v62, v69, v225 :: v_dual_fmac_f32 v63, v79, v225
	v_dual_fmac_f32 v72, v89, v225 :: v_dual_fmac_f32 v73, v99, v225
	v_dual_fmac_f32 v82, v109, v225 :: v_dual_fmac_f32 v83, v119, v225
	v_dual_fmac_f32 v92, v49, v233 :: v_dual_fmac_f32 v93, v59, v233
	v_dual_fmac_f32 v102, v69, v233 :: v_dual_fmac_f32 v103, v79, v233
	v_dual_fmac_f32 v112, v89, v233 :: v_dual_fmac_f32 v113, v99, v233
	v_dual_fmac_f32 v122, v109, v233 :: v_dual_fmac_f32 v123, v119, v233
	v_dual_fmac_f32 v52, v50, v226 :: v_dual_fmac_f32 v53, v60, v226
	v_dual_fmac_f32 v62, v70, v226 :: v_dual_fmac_f32 v63, v80, v226
	v_dual_fmac_f32 v72, v90, v226 :: v_dual_fmac_f32 v73, v100, v226
	v_dual_fmac_f32 v82, v110, v226 :: v_dual_fmac_f32 v83, v120, v226
	v_dual_fmac_f32 v92, v50, v234 :: v_dual_fmac_f32 v93, v60, v234
	v_dual_fmac_f32 v102, v70, v234 :: v_dual_fmac_f32 v103, v80, v234
	v_dual_fmac_f32 v112, v90, v234 :: v_dual_fmac_f32 v113, v100, v234
	v_dual_fmac_f32 v122, v110, v234 :: v_dual_fmac_f32 v123, v120, v234
	v_dual_fmac_f32 v52, v51, v227 :: v_dual_fmac_f32 v53, v61, v227
	v_dual_fmac_f32 v62, v71, v227 :: v_dual_fmac_f32 v63, v81, v227
	v_dual_fmac_f32 v72, v91, v227 :: v_dual_fmac_f32 v73, v101, v227
	v_dual_fmac_f32 v82, v111, v227 :: v_dual_fmac_f32 v83, v121, v227
	v_dual_fmac_f32 v92, v51, v235 :: v_dual_fmac_f32 v93, v61, v235
	v_dual_fmac_f32 v102, v71, v235 :: v_dual_fmac_f32 v103, v81, v235
	v_dual_fmac_f32 v112, v91, v235 :: v_dual_fmac_f32 v113, v101, v235
	v_dual_fmac_f32 v122, v111, v235 :: v_dual_fmac_f32 v123, v121, v235
	global_load_b128 v[220:223], v3, s[28:29]
	global_load_b128 v[224:227], v3, s[28:29] offset:16
	global_load_b128 v[228:231], v3, s[30:31]
	global_load_b128 v[232:235], v3, s[30:31] offset:16
	v_dual_add_f32 v140, v52, v140 :: v_dual_add_f32 v141, v141, v53
	v_dual_add_f32 v142, v62, v142 :: v_dual_add_f32 v143, v143, v63
	v_dual_add_f32 v144, v72, v144 :: v_dual_add_f32 v145, v145, v73
	v_dual_add_f32 v146, v82, v146 :: v_dual_add_f32 v147, v147, v83
	v_dual_add_f32 v164, v92, v164 :: v_dual_add_f32 v165, v165, v93
	v_dual_add_f32 v166, v102, v166 :: v_dual_add_f32 v167, v167, v103
	v_dual_add_f32 v168, v112, v168 :: v_dual_add_f32 v169, v169, v113
	v_dual_add_f32 v170, v122, v170 :: v_dual_add_f32 v171, v171, v123
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v52, v44, v236 :: v_dual_mul_f32 v53, v55, v237
	v_dual_mul_f32 v62, v64, v236 :: v_dual_mul_f32 v63, v75, v237
	v_dual_mul_f32 v72, v84, v236 :: v_dual_mul_f32 v73, v95, v237
	v_dual_mul_f32 v82, v104, v236 :: v_dual_mul_f32 v83, v115, v237
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v92, v44, v244 :: v_dual_mul_f32 v93, v55, v245
	v_dual_mul_f32 v102, v64, v244 :: v_dual_mul_f32 v103, v75, v245
	v_dual_mul_f32 v112, v84, v244 :: v_dual_mul_f32 v113, v95, v245
	v_dual_mul_f32 v122, v104, v244 :: v_dual_mul_f32 v123, v115, v245
	v_dual_fmac_f32 v52, v45, v237 :: v_dual_fmac_f32 v53, v54, v236
	v_dual_fmac_f32 v62, v65, v237 :: v_dual_fmac_f32 v63, v74, v236
	v_dual_fmac_f32 v72, v85, v237 :: v_dual_fmac_f32 v73, v94, v236
	v_dual_fmac_f32 v82, v105, v237 :: v_dual_fmac_f32 v83, v114, v236
	v_dual_fmac_f32 v92, v45, v245 :: v_dual_fmac_f32 v93, v54, v244
	v_dual_fmac_f32 v102, v65, v245 :: v_dual_fmac_f32 v103, v74, v244
	v_dual_fmac_f32 v112, v85, v245 :: v_dual_fmac_f32 v113, v94, v244
	v_dual_fmac_f32 v122, v105, v245 :: v_dual_fmac_f32 v123, v114, v244
	v_dual_fmac_f32 v52, v46, v238 :: v_dual_fmac_f32 v53, v56, v238
	v_dual_fmac_f32 v62, v66, v238 :: v_dual_fmac_f32 v63, v76, v238
	v_dual_fmac_f32 v72, v86, v238 :: v_dual_fmac_f32 v73, v96, v238
	v_dual_fmac_f32 v82, v106, v238 :: v_dual_fmac_f32 v83, v116, v238
	v_dual_fmac_f32 v92, v46, v246 :: v_dual_fmac_f32 v93, v56, v246
	v_dual_fmac_f32 v102, v66, v246 :: v_dual_fmac_f32 v103, v76, v246
	v_dual_fmac_f32 v112, v86, v246 :: v_dual_fmac_f32 v113, v96, v246
	v_dual_fmac_f32 v122, v106, v246 :: v_dual_fmac_f32 v123, v116, v246
	v_dual_fmac_f32 v52, v47, v239 :: v_dual_fmac_f32 v53, v57, v239
	v_dual_fmac_f32 v62, v67, v239 :: v_dual_fmac_f32 v63, v77, v239
	v_dual_fmac_f32 v72, v87, v239 :: v_dual_fmac_f32 v73, v97, v239
	v_dual_fmac_f32 v82, v107, v239 :: v_dual_fmac_f32 v83, v117, v239
	v_dual_fmac_f32 v92, v47, v247 :: v_dual_fmac_f32 v93, v57, v247
	v_dual_fmac_f32 v102, v67, v247 :: v_dual_fmac_f32 v103, v77, v247
	v_dual_fmac_f32 v112, v87, v247 :: v_dual_fmac_f32 v113, v97, v247
	v_dual_fmac_f32 v122, v107, v247 :: v_dual_fmac_f32 v123, v117, v247
	v_dual_fmac_f32 v52, v48, v240 :: v_dual_fmac_f32 v53, v58, v240
	v_dual_fmac_f32 v62, v68, v240 :: v_dual_fmac_f32 v63, v78, v240
	v_dual_fmac_f32 v72, v88, v240 :: v_dual_fmac_f32 v73, v98, v240
	v_dual_fmac_f32 v82, v108, v240 :: v_dual_fmac_f32 v83, v118, v240
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v92, v48, v248 :: v_dual_fmac_f32 v93, v58, v248
	v_dual_fmac_f32 v102, v68, v248 :: v_dual_fmac_f32 v103, v78, v248
	v_dual_fmac_f32 v112, v88, v248 :: v_dual_fmac_f32 v113, v98, v248
	v_dual_fmac_f32 v122, v108, v248 :: v_dual_fmac_f32 v123, v118, v248
	v_dual_fmac_f32 v52, v49, v241 :: v_dual_fmac_f32 v53, v59, v241
	v_dual_fmac_f32 v62, v69, v241 :: v_dual_fmac_f32 v63, v79, v241
	v_dual_fmac_f32 v72, v89, v241 :: v_dual_fmac_f32 v73, v99, v241
	v_dual_fmac_f32 v82, v109, v241 :: v_dual_fmac_f32 v83, v119, v241
	v_dual_fmac_f32 v92, v49, v249 :: v_dual_fmac_f32 v93, v59, v249
	v_dual_fmac_f32 v102, v69, v249 :: v_dual_fmac_f32 v103, v79, v249
	v_dual_fmac_f32 v112, v89, v249 :: v_dual_fmac_f32 v113, v99, v249
	v_dual_fmac_f32 v122, v109, v249 :: v_dual_fmac_f32 v123, v119, v249
	v_dual_fmac_f32 v52, v50, v242 :: v_dual_fmac_f32 v53, v60, v242
	v_dual_fmac_f32 v62, v70, v242 :: v_dual_fmac_f32 v63, v80, v242
	v_dual_fmac_f32 v72, v90, v242 :: v_dual_fmac_f32 v73, v100, v242
	v_dual_fmac_f32 v82, v110, v242 :: v_dual_fmac_f32 v83, v120, v242
	v_dual_fmac_f32 v92, v50, v250 :: v_dual_fmac_f32 v93, v60, v250
	v_dual_fmac_f32 v102, v70, v250 :: v_dual_fmac_f32 v103, v80, v250
	v_dual_fmac_f32 v112, v90, v250 :: v_dual_fmac_f32 v113, v100, v250
	v_dual_fmac_f32 v122, v110, v250 :: v_dual_fmac_f32 v123, v120, v250
	v_dual_fmac_f32 v52, v51, v243 :: v_dual_fmac_f32 v53, v61, v243
	v_dual_fmac_f32 v62, v71, v243 :: v_dual_fmac_f32 v63, v81, v243
	v_dual_fmac_f32 v72, v91, v243 :: v_dual_fmac_f32 v73, v101, v243
	v_dual_fmac_f32 v82, v111, v243 :: v_dual_fmac_f32 v83, v121, v243
	v_dual_fmac_f32 v92, v51, v251 :: v_dual_fmac_f32 v93, v61, v251
	v_dual_fmac_f32 v102, v71, v251 :: v_dual_fmac_f32 v103, v81, v251
	v_dual_fmac_f32 v112, v91, v251 :: v_dual_fmac_f32 v113, v101, v251
	v_dual_fmac_f32 v122, v111, v251 :: v_dual_fmac_f32 v123, v121, v251
	global_load_b128 v[236:239], v3, s[32:33]
	global_load_b128 v[240:243], v3, s[32:33] offset:16
	global_load_b128 v[244:247], v3, s[34:35]
	global_load_b128 v[248:251], v3, s[34:35] offset:16
	v_dual_add_f32 v188, v52, v188 :: v_dual_add_f32 v189, v189, v53
	v_dual_add_f32 v190, v62, v190 :: v_dual_add_f32 v191, v191, v63
	v_dual_add_f32 v192, v72, v192 :: v_dual_add_f32 v193, v193, v73
	v_dual_add_f32 v194, v82, v194 :: v_dual_add_f32 v195, v195, v83
	v_dual_add_f32 v212, v92, v212 :: v_dual_add_f32 v213, v213, v93
	v_dual_add_f32 v214, v102, v214 :: v_dual_add_f32 v215, v215, v103
	v_dual_add_f32 v216, v112, v216 :: v_dual_add_f32 v217, v217, v113
	v_dual_add_f32 v218, v122, v218 :: v_dual_add_f32 v219, v219, v123
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s3
	s_wait_loadcnt 0x0
	.Lrx_b4_s3_done:
	v_dual_add_f32 v132, v132, v140 :: v_dual_add_f32 v133, v133, v141
	v_dual_add_f32 v134, v134, v142 :: v_dual_add_f32 v135, v135, v143
	v_dual_add_f32 v136, v136, v144 :: v_dual_add_f32 v137, v137, v145
	v_dual_add_f32 v138, v138, v146 :: v_dual_add_f32 v139, v139, v147
	v_dual_add_f32 v156, v156, v164 :: v_dual_add_f32 v157, v157, v165
	v_dual_add_f32 v158, v158, v166 :: v_dual_add_f32 v159, v159, v167
	v_dual_add_f32 v160, v160, v168 :: v_dual_add_f32 v161, v161, v169
	v_dual_add_f32 v162, v162, v170 :: v_dual_add_f32 v163, v163, v171
	v_dual_add_f32 v180, v180, v188 :: v_dual_add_f32 v181, v181, v189
	v_dual_add_f32 v182, v182, v190 :: v_dual_add_f32 v183, v183, v191
	v_dual_add_f32 v184, v184, v192 :: v_dual_add_f32 v185, v185, v193
	v_dual_add_f32 v186, v186, v194 :: v_dual_add_f32 v187, v187, v195
	v_dual_add_f32 v204, v204, v212 :: v_dual_add_f32 v205, v205, v213
	v_dual_add_f32 v206, v206, v214 :: v_dual_add_f32 v207, v207, v215
	v_dual_add_f32 v208, v208, v216 :: v_dual_add_f32 v209, v209, v217
	v_dual_add_f32 v210, v210, v218 :: v_dual_add_f32 v211, v211, v219
	v_dual_add_f32 v124, v124, v132 :: v_dual_add_f32 v125, v125, v133
	v_dual_add_f32 v126, v126, v134 :: v_dual_add_f32 v127, v127, v135
	v_dual_add_f32 v128, v128, v136 :: v_dual_add_f32 v129, v129, v137
	v_dual_add_f32 v130, v130, v138 :: v_dual_add_f32 v131, v131, v139
	v_dual_add_f32 v148, v148, v156 :: v_dual_add_f32 v149, v149, v157
	v_dual_add_f32 v150, v150, v158 :: v_dual_add_f32 v151, v151, v159
	v_dual_add_f32 v152, v152, v160 :: v_dual_add_f32 v153, v153, v161
	v_dual_add_f32 v154, v154, v162 :: v_dual_add_f32 v155, v155, v163
	v_dual_add_f32 v172, v172, v180 :: v_dual_add_f32 v173, v173, v181
	v_dual_add_f32 v174, v174, v182 :: v_dual_add_f32 v175, v175, v183
	v_dual_add_f32 v176, v176, v184 :: v_dual_add_f32 v177, v177, v185
	v_dual_add_f32 v178, v178, v186 :: v_dual_add_f32 v179, v179, v187
	v_dual_add_f32 v196, v196, v204 :: v_dual_add_f32 v197, v197, v205
	v_dual_add_f32 v198, v198, v206 :: v_dual_add_f32 v199, v199, v207
	v_dual_add_f32 v200, v200, v208 :: v_dual_add_f32 v201, v201, v209
	v_dual_add_f32 v202, v202, v210 :: v_dual_add_f32 v203, v203, v211
	ds_swizzle_b32 v220, v124 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v221, v125 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v222, v126 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v223, v127 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v224, v128 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v225, v129 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v226, v130 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v227, v131 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v228, v148 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v229, v149 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v230, v150 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v231, v151 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v232, v152 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v233, v153 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v234, v154 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v235, v155 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v236, v172 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v237, v173 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v238, v174 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v239, v175 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v240, v176 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v241, v177 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v242, v178 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v243, v179 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v244, v196 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v245, v197 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v246, v198 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v247, v199 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v248, v200 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v249, v201 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v250, v202 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v251, v203 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x1f
	v_add_f32_e32 v124, v124, v220
	s_wait_dscnt 0x1e
	v_add_f32_e32 v125, v125, v221
	s_wait_dscnt 0x1d
	v_add_f32_e32 v126, v126, v222
	s_wait_dscnt 0x1c
	v_add_f32_e32 v127, v127, v223
	s_wait_dscnt 0x1b
	v_add_f32_e32 v128, v128, v224
	s_wait_dscnt 0x1a
	v_add_f32_e32 v129, v129, v225
	s_wait_dscnt 0x19
	v_add_f32_e32 v130, v130, v226
	s_wait_dscnt 0x18
	v_add_f32_e32 v131, v131, v227
	s_wait_dscnt 0x17
	v_add_f32_e32 v148, v148, v228
	s_wait_dscnt 0x16
	v_add_f32_e32 v149, v149, v229
	s_wait_dscnt 0x15
	v_add_f32_e32 v150, v150, v230
	s_wait_dscnt 0x14
	v_add_f32_e32 v151, v151, v231
	s_wait_dscnt 0x13
	v_add_f32_e32 v152, v152, v232
	s_wait_dscnt 0x12
	v_add_f32_e32 v153, v153, v233
	s_wait_dscnt 0x11
	v_add_f32_e32 v154, v154, v234
	s_wait_dscnt 0x10
	v_add_f32_e32 v155, v155, v235
	s_wait_dscnt 0xf
	v_add_f32_e32 v172, v172, v236
	s_wait_dscnt 0xe
	v_add_f32_e32 v173, v173, v237
	s_wait_dscnt 0xd
	v_add_f32_e32 v174, v174, v238
	s_wait_dscnt 0xc
	v_add_f32_e32 v175, v175, v239
	s_wait_dscnt 0xb
	v_add_f32_e32 v176, v176, v240
	s_wait_dscnt 0xa
	v_add_f32_e32 v177, v177, v241
	s_wait_dscnt 0x9
	v_add_f32_e32 v178, v178, v242
	s_wait_dscnt 0x8
	v_add_f32_e32 v179, v179, v243
	s_wait_dscnt 0x7
	v_add_f32_e32 v196, v196, v244
	s_wait_dscnt 0x6
	v_add_f32_e32 v197, v197, v245
	s_wait_dscnt 0x5
	v_add_f32_e32 v198, v198, v246
	s_wait_dscnt 0x4
	v_add_f32_e32 v199, v199, v247
	s_wait_dscnt 0x3
	v_add_f32_e32 v200, v200, v248
	s_wait_dscnt 0x2
	v_add_f32_e32 v201, v201, v249
	s_wait_dscnt 0x1
	v_add_f32_e32 v202, v202, v250
	s_wait_dscnt 0x0
	v_add_f32_e32 v203, v203, v251
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v220, v3, v124
	ds_bpermute_b32 v221, v3, v125
	ds_bpermute_b32 v222, v3, v126
	ds_bpermute_b32 v223, v3, v127
	ds_bpermute_b32 v224, v3, v128
	ds_bpermute_b32 v225, v3, v129
	ds_bpermute_b32 v226, v3, v130
	ds_bpermute_b32 v227, v3, v131
	ds_bpermute_b32 v228, v3, v148
	ds_bpermute_b32 v229, v3, v149
	ds_bpermute_b32 v230, v3, v150
	ds_bpermute_b32 v231, v3, v151
	ds_bpermute_b32 v232, v3, v152
	ds_bpermute_b32 v233, v3, v153
	ds_bpermute_b32 v234, v3, v154
	ds_bpermute_b32 v235, v3, v155
	ds_bpermute_b32 v236, v3, v172
	ds_bpermute_b32 v237, v3, v173
	ds_bpermute_b32 v238, v3, v174
	ds_bpermute_b32 v239, v3, v175
	ds_bpermute_b32 v240, v3, v176
	ds_bpermute_b32 v241, v3, v177
	ds_bpermute_b32 v242, v3, v178
	ds_bpermute_b32 v243, v3, v179
	ds_bpermute_b32 v244, v3, v196
	ds_bpermute_b32 v245, v3, v197
	ds_bpermute_b32 v246, v3, v198
	ds_bpermute_b32 v247, v3, v199
	ds_bpermute_b32 v248, v3, v200
	ds_bpermute_b32 v249, v3, v201
	ds_bpermute_b32 v250, v3, v202
	ds_bpermute_b32 v251, v3, v203
	s_wait_dscnt 0x1f
	v_add_f32_e32 v124, v124, v220
	s_wait_dscnt 0x1e
	v_add_f32_e32 v125, v125, v221
	s_wait_dscnt 0x1d
	v_add_f32_e32 v126, v126, v222
	s_wait_dscnt 0x1c
	v_add_f32_e32 v127, v127, v223
	s_wait_dscnt 0x1b
	v_add_f32_e32 v128, v128, v224
	s_wait_dscnt 0x1a
	v_add_f32_e32 v129, v129, v225
	s_wait_dscnt 0x19
	v_add_f32_e32 v130, v130, v226
	s_wait_dscnt 0x18
	v_add_f32_e32 v131, v131, v227
	s_wait_dscnt 0x17
	v_add_f32_e32 v148, v148, v228
	s_wait_dscnt 0x16
	v_add_f32_e32 v149, v149, v229
	s_wait_dscnt 0x15
	v_add_f32_e32 v150, v150, v230
	s_wait_dscnt 0x14
	v_add_f32_e32 v151, v151, v231
	s_wait_dscnt 0x13
	v_add_f32_e32 v152, v152, v232
	s_wait_dscnt 0x12
	v_add_f32_e32 v153, v153, v233
	s_wait_dscnt 0x11
	v_add_f32_e32 v154, v154, v234
	s_wait_dscnt 0x10
	v_add_f32_e32 v155, v155, v235
	s_wait_dscnt 0xf
	v_add_f32_e32 v172, v172, v236
	s_wait_dscnt 0xe
	v_add_f32_e32 v173, v173, v237
	s_wait_dscnt 0xd
	v_add_f32_e32 v174, v174, v238
	s_wait_dscnt 0xc
	v_add_f32_e32 v175, v175, v239
	s_wait_dscnt 0xb
	v_add_f32_e32 v176, v176, v240
	s_wait_dscnt 0xa
	v_add_f32_e32 v177, v177, v241
	s_wait_dscnt 0x9
	v_add_f32_e32 v178, v178, v242
	s_wait_dscnt 0x8
	v_add_f32_e32 v179, v179, v243
	s_wait_dscnt 0x7
	v_add_f32_e32 v196, v196, v244
	s_wait_dscnt 0x6
	v_add_f32_e32 v197, v197, v245
	s_wait_dscnt 0x5
	v_add_f32_e32 v198, v198, v246
	s_wait_dscnt 0x4
	v_add_f32_e32 v199, v199, v247
	s_wait_dscnt 0x3
	v_add_f32_e32 v200, v200, v248
	s_wait_dscnt 0x2
	v_add_f32_e32 v201, v201, v249
	s_wait_dscnt 0x1
	v_add_f32_e32 v202, v202, v250
	s_wait_dscnt 0x0
	v_add_f32_e32 v203, v203, v251
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v220, v3, v124
	ds_bpermute_b32 v221, v3, v125
	ds_bpermute_b32 v222, v3, v126
	ds_bpermute_b32 v223, v3, v127
	ds_bpermute_b32 v224, v3, v128
	ds_bpermute_b32 v225, v3, v129
	ds_bpermute_b32 v226, v3, v130
	ds_bpermute_b32 v227, v3, v131
	ds_bpermute_b32 v228, v3, v148
	ds_bpermute_b32 v229, v3, v149
	ds_bpermute_b32 v230, v3, v150
	ds_bpermute_b32 v231, v3, v151
	ds_bpermute_b32 v232, v3, v152
	ds_bpermute_b32 v233, v3, v153
	ds_bpermute_b32 v234, v3, v154
	ds_bpermute_b32 v235, v3, v155
	ds_bpermute_b32 v236, v3, v172
	ds_bpermute_b32 v237, v3, v173
	ds_bpermute_b32 v238, v3, v174
	ds_bpermute_b32 v239, v3, v175
	ds_bpermute_b32 v240, v3, v176
	ds_bpermute_b32 v241, v3, v177
	ds_bpermute_b32 v242, v3, v178
	ds_bpermute_b32 v243, v3, v179
	ds_bpermute_b32 v244, v3, v196
	ds_bpermute_b32 v245, v3, v197
	ds_bpermute_b32 v246, v3, v198
	ds_bpermute_b32 v247, v3, v199
	ds_bpermute_b32 v248, v3, v200
	ds_bpermute_b32 v249, v3, v201
	ds_bpermute_b32 v250, v3, v202
	ds_bpermute_b32 v251, v3, v203
	s_wait_dscnt 0x1f
	v_add_f32_e32 v124, v124, v220
	s_wait_dscnt 0x1e
	v_add_f32_e32 v125, v125, v221
	s_wait_dscnt 0x1d
	v_add_f32_e32 v126, v126, v222
	s_wait_dscnt 0x1c
	v_add_f32_e32 v127, v127, v223
	s_wait_dscnt 0x1b
	v_add_f32_e32 v128, v128, v224
	s_wait_dscnt 0x1a
	v_add_f32_e32 v129, v129, v225
	s_wait_dscnt 0x19
	v_add_f32_e32 v130, v130, v226
	s_wait_dscnt 0x18
	v_add_f32_e32 v131, v131, v227
	s_wait_dscnt 0x17
	v_add_f32_e32 v148, v148, v228
	s_wait_dscnt 0x16
	v_add_f32_e32 v149, v149, v229
	s_wait_dscnt 0x15
	v_add_f32_e32 v150, v150, v230
	s_wait_dscnt 0x14
	v_add_f32_e32 v151, v151, v231
	s_wait_dscnt 0x13
	v_add_f32_e32 v152, v152, v232
	s_wait_dscnt 0x12
	v_add_f32_e32 v153, v153, v233
	s_wait_dscnt 0x11
	v_add_f32_e32 v154, v154, v234
	s_wait_dscnt 0x10
	v_add_f32_e32 v155, v155, v235
	s_wait_dscnt 0xf
	v_add_f32_e32 v172, v172, v236
	s_wait_dscnt 0xe
	v_add_f32_e32 v173, v173, v237
	s_wait_dscnt 0xd
	v_add_f32_e32 v174, v174, v238
	s_wait_dscnt 0xc
	v_add_f32_e32 v175, v175, v239
	s_wait_dscnt 0xb
	v_add_f32_e32 v176, v176, v240
	s_wait_dscnt 0xa
	v_add_f32_e32 v177, v177, v241
	s_wait_dscnt 0x9
	v_add_f32_e32 v178, v178, v242
	s_wait_dscnt 0x8
	v_add_f32_e32 v179, v179, v243
	s_wait_dscnt 0x7
	v_add_f32_e32 v196, v196, v244
	s_wait_dscnt 0x6
	v_add_f32_e32 v197, v197, v245
	s_wait_dscnt 0x5
	v_add_f32_e32 v198, v198, v246
	s_wait_dscnt 0x4
	v_add_f32_e32 v199, v199, v247
	s_wait_dscnt 0x3
	v_add_f32_e32 v200, v200, v248
	s_wait_dscnt 0x2
	v_add_f32_e32 v201, v201, v249
	s_wait_dscnt 0x1
	v_add_f32_e32 v202, v202, v250
	s_wait_dscnt 0x0
	v_add_f32_e32 v203, v203, v251
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v220, v3, v124
	ds_bpermute_b32 v221, v3, v125
	ds_bpermute_b32 v222, v3, v126
	ds_bpermute_b32 v223, v3, v127
	ds_bpermute_b32 v224, v3, v128
	ds_bpermute_b32 v225, v3, v129
	ds_bpermute_b32 v226, v3, v130
	ds_bpermute_b32 v227, v3, v131
	ds_bpermute_b32 v228, v3, v148
	ds_bpermute_b32 v229, v3, v149
	ds_bpermute_b32 v230, v3, v150
	ds_bpermute_b32 v231, v3, v151
	ds_bpermute_b32 v232, v3, v152
	ds_bpermute_b32 v233, v3, v153
	ds_bpermute_b32 v234, v3, v154
	ds_bpermute_b32 v235, v3, v155
	ds_bpermute_b32 v236, v3, v172
	ds_bpermute_b32 v237, v3, v173
	ds_bpermute_b32 v238, v3, v174
	ds_bpermute_b32 v239, v3, v175
	ds_bpermute_b32 v240, v3, v176
	ds_bpermute_b32 v241, v3, v177
	ds_bpermute_b32 v242, v3, v178
	ds_bpermute_b32 v243, v3, v179
	ds_bpermute_b32 v244, v3, v196
	ds_bpermute_b32 v245, v3, v197
	ds_bpermute_b32 v246, v3, v198
	ds_bpermute_b32 v247, v3, v199
	ds_bpermute_b32 v248, v3, v200
	ds_bpermute_b32 v249, v3, v201
	ds_bpermute_b32 v250, v3, v202
	ds_bpermute_b32 v251, v3, v203
	s_wait_dscnt 0x1f
	v_add_f32_e32 v124, v124, v220
	s_wait_dscnt 0x1e
	v_add_f32_e32 v125, v125, v221
	s_wait_dscnt 0x1d
	v_add_f32_e32 v126, v126, v222
	s_wait_dscnt 0x1c
	v_add_f32_e32 v127, v127, v223
	s_wait_dscnt 0x1b
	v_add_f32_e32 v128, v128, v224
	s_wait_dscnt 0x1a
	v_add_f32_e32 v129, v129, v225
	s_wait_dscnt 0x19
	v_add_f32_e32 v130, v130, v226
	s_wait_dscnt 0x18
	v_add_f32_e32 v131, v131, v227
	s_wait_dscnt 0x17
	v_add_f32_e32 v148, v148, v228
	s_wait_dscnt 0x16
	v_add_f32_e32 v149, v149, v229
	s_wait_dscnt 0x15
	v_add_f32_e32 v150, v150, v230
	s_wait_dscnt 0x14
	v_add_f32_e32 v151, v151, v231
	s_wait_dscnt 0x13
	v_add_f32_e32 v152, v152, v232
	s_wait_dscnt 0x12
	v_add_f32_e32 v153, v153, v233
	s_wait_dscnt 0x11
	v_add_f32_e32 v154, v154, v234
	s_wait_dscnt 0x10
	v_add_f32_e32 v155, v155, v235
	s_wait_dscnt 0xf
	v_add_f32_e32 v172, v172, v236
	s_wait_dscnt 0xe
	v_add_f32_e32 v173, v173, v237
	s_wait_dscnt 0xd
	v_add_f32_e32 v174, v174, v238
	s_wait_dscnt 0xc
	v_add_f32_e32 v175, v175, v239
	s_wait_dscnt 0xb
	v_add_f32_e32 v176, v176, v240
	s_wait_dscnt 0xa
	v_add_f32_e32 v177, v177, v241
	s_wait_dscnt 0x9
	v_add_f32_e32 v178, v178, v242
	s_wait_dscnt 0x8
	v_add_f32_e32 v179, v179, v243
	s_wait_dscnt 0x7
	v_add_f32_e32 v196, v196, v244
	s_wait_dscnt 0x6
	v_add_f32_e32 v197, v197, v245
	s_wait_dscnt 0x5
	v_add_f32_e32 v198, v198, v246
	s_wait_dscnt 0x4
	v_add_f32_e32 v199, v199, v247
	s_wait_dscnt 0x3
	v_add_f32_e32 v200, v200, v248
	s_wait_dscnt 0x2
	v_add_f32_e32 v201, v201, v249
	s_wait_dscnt 0x1
	v_add_f32_e32 v202, v202, v250
	s_wait_dscnt 0x0
	v_add_f32_e32 v203, v203, v251
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v220, v3, v124
	ds_bpermute_b32 v221, v3, v125
	ds_bpermute_b32 v222, v3, v126
	ds_bpermute_b32 v223, v3, v127
	ds_bpermute_b32 v224, v3, v128
	ds_bpermute_b32 v225, v3, v129
	ds_bpermute_b32 v226, v3, v130
	ds_bpermute_b32 v227, v3, v131
	ds_bpermute_b32 v228, v3, v148
	ds_bpermute_b32 v229, v3, v149
	ds_bpermute_b32 v230, v3, v150
	ds_bpermute_b32 v231, v3, v151
	ds_bpermute_b32 v232, v3, v152
	ds_bpermute_b32 v233, v3, v153
	ds_bpermute_b32 v234, v3, v154
	ds_bpermute_b32 v235, v3, v155
	ds_bpermute_b32 v236, v3, v172
	ds_bpermute_b32 v237, v3, v173
	ds_bpermute_b32 v238, v3, v174
	ds_bpermute_b32 v239, v3, v175
	ds_bpermute_b32 v240, v3, v176
	ds_bpermute_b32 v241, v3, v177
	ds_bpermute_b32 v242, v3, v178
	ds_bpermute_b32 v243, v3, v179
	ds_bpermute_b32 v244, v3, v196
	ds_bpermute_b32 v245, v3, v197
	ds_bpermute_b32 v246, v3, v198
	ds_bpermute_b32 v247, v3, v199
	ds_bpermute_b32 v248, v3, v200
	ds_bpermute_b32 v249, v3, v201
	ds_bpermute_b32 v250, v3, v202
	ds_bpermute_b32 v251, v3, v203
	s_wait_dscnt 0x1f
	v_add_f32_e32 v124, v124, v220
	s_wait_dscnt 0x1e
	v_add_f32_e32 v125, v125, v221
	s_wait_dscnt 0x1d
	v_add_f32_e32 v126, v126, v222
	s_wait_dscnt 0x1c
	v_add_f32_e32 v127, v127, v223
	s_wait_dscnt 0x1b
	v_add_f32_e32 v128, v128, v224
	s_wait_dscnt 0x1a
	v_add_f32_e32 v129, v129, v225
	s_wait_dscnt 0x19
	v_add_f32_e32 v130, v130, v226
	s_wait_dscnt 0x18
	v_add_f32_e32 v131, v131, v227
	s_wait_dscnt 0x17
	v_add_f32_e32 v148, v148, v228
	s_wait_dscnt 0x16
	v_add_f32_e32 v149, v149, v229
	s_wait_dscnt 0x15
	v_add_f32_e32 v150, v150, v230
	s_wait_dscnt 0x14
	v_add_f32_e32 v151, v151, v231
	s_wait_dscnt 0x13
	v_add_f32_e32 v152, v152, v232
	s_wait_dscnt 0x12
	v_add_f32_e32 v153, v153, v233
	s_wait_dscnt 0x11
	v_add_f32_e32 v154, v154, v234
	s_wait_dscnt 0x10
	v_add_f32_e32 v155, v155, v235
	s_wait_dscnt 0xf
	v_add_f32_e32 v172, v172, v236
	s_wait_dscnt 0xe
	v_add_f32_e32 v173, v173, v237
	s_wait_dscnt 0xd
	v_add_f32_e32 v174, v174, v238
	s_wait_dscnt 0xc
	v_add_f32_e32 v175, v175, v239
	s_wait_dscnt 0xb
	v_add_f32_e32 v176, v176, v240
	s_wait_dscnt 0xa
	v_add_f32_e32 v177, v177, v241
	s_wait_dscnt 0x9
	v_add_f32_e32 v178, v178, v242
	s_wait_dscnt 0x8
	v_add_f32_e32 v179, v179, v243
	s_wait_dscnt 0x7
	v_add_f32_e32 v196, v196, v244
	s_wait_dscnt 0x6
	v_add_f32_e32 v197, v197, v245
	s_wait_dscnt 0x5
	v_add_f32_e32 v198, v198, v246
	s_wait_dscnt 0x4
	v_add_f32_e32 v199, v199, v247
	s_wait_dscnt 0x3
	v_add_f32_e32 v200, v200, v248
	s_wait_dscnt 0x2
	v_add_f32_e32 v201, v201, v249
	s_wait_dscnt 0x1
	v_add_f32_e32 v202, v202, v250
	s_wait_dscnt 0x0
	v_add_f32_e32 v203, v203, v251
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v76, s46
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v77, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v78, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v79, s47
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b4_p0_single
	global_load_b32 v44, v76, s[8:9] offset:0
	global_load_b32 v45, v76, s[8:9] offset:4
	global_load_b32 v52, v77, s[8:9] offset:0
	global_load_b32 v53, v77, s[8:9] offset:4
	global_load_b32 v60, v78, s[8:9] offset:0
	global_load_b32 v61, v78, s[8:9] offset:4
	global_load_b32 v68, v79, s[8:9] offset:0
	global_load_b32 v69, v79, s[8:9] offset:4
	s_wait_loadcnt 0x7
	v_add_f32_e32 v44, v124, v44
	global_store_b32 v76, v44, s[8:9] offset:0
	s_wait_loadcnt 0x6
	v_add_f32_e32 v45, v125, v45
	global_store_b32 v76, v45, s[8:9] offset:4
	s_wait_loadcnt 0x5
	v_add_f32_e32 v52, v148, v52
	global_store_b32 v77, v52, s[8:9] offset:0
	s_wait_loadcnt 0x4
	v_add_f32_e32 v53, v149, v53
	global_store_b32 v77, v53, s[8:9] offset:4
	s_wait_loadcnt 0x3
	v_add_f32_e32 v60, v172, v60
	global_store_b32 v78, v60, s[8:9] offset:0
	s_wait_loadcnt 0x2
	v_add_f32_e32 v61, v173, v61
	global_store_b32 v78, v61, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v68, v196, v68
	global_store_b32 v79, v68, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v69, v197, v69
	global_store_b32 v79, v69, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b4_p0_next
	.Lrx_b4_p0_single:
	global_load_b32 v44, v76, s[8:9] offset:0
	global_load_b32 v52, v77, s[8:9] offset:0
	global_load_b32 v60, v78, s[8:9] offset:0
	global_load_b32 v68, v79, s[8:9] offset:0
	s_wait_loadcnt 0x3
	v_add_f32_e32 v44, v44, v124
	global_store_b32 v76, v44, s[8:9] offset:0
	s_wait_loadcnt 0x2
	v_add_f32_e32 v52, v52, v148
	global_store_b32 v77, v52, s[8:9] offset:0
	s_wait_loadcnt 0x1
	v_add_f32_e32 v60, v60, v172
	global_store_b32 v78, v60, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v68, v68, v196
	global_store_b32 v79, v68, s[8:9] offset:0
	s_wait_storecnt 0x0
	s_branch .Lrx_b4_stored
	.Lrx_b4_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b4_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b4_p1_single
	global_load_b32 v46, v76, s[8:9] offset:8
	global_load_b32 v47, v76, s[8:9] offset:12
	global_load_b32 v54, v77, s[8:9] offset:8
	global_load_b32 v55, v77, s[8:9] offset:12
	global_load_b32 v62, v78, s[8:9] offset:8
	global_load_b32 v63, v78, s[8:9] offset:12
	global_load_b32 v70, v79, s[8:9] offset:8
	global_load_b32 v71, v79, s[8:9] offset:12
	s_wait_loadcnt 0x7
	v_add_f32_e32 v46, v126, v46
	global_store_b32 v76, v46, s[8:9] offset:8
	s_wait_loadcnt 0x6
	v_add_f32_e32 v47, v127, v47
	global_store_b32 v76, v47, s[8:9] offset:12
	s_wait_loadcnt 0x5
	v_add_f32_e32 v54, v150, v54
	global_store_b32 v77, v54, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v55, v151, v55
	global_store_b32 v77, v55, s[8:9] offset:12
	s_wait_loadcnt 0x3
	v_add_f32_e32 v62, v174, v62
	global_store_b32 v78, v62, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v63, v175, v63
	global_store_b32 v78, v63, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v70, v198, v70
	global_store_b32 v79, v70, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v71, v199, v71
	global_store_b32 v79, v71, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b4_p1_next
	.Lrx_b4_p1_single:
	global_load_b32 v46, v76, s[8:9] offset:8
	global_load_b32 v54, v77, s[8:9] offset:8
	global_load_b32 v62, v78, s[8:9] offset:8
	global_load_b32 v70, v79, s[8:9] offset:8
	s_wait_loadcnt 0x3
	v_add_f32_e32 v46, v46, v126
	global_store_b32 v76, v46, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v54, v54, v150
	global_store_b32 v77, v54, s[8:9] offset:8
	s_wait_loadcnt 0x1
	v_add_f32_e32 v62, v62, v174
	global_store_b32 v78, v62, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v70, v70, v198
	global_store_b32 v79, v70, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b4_stored
	.Lrx_b4_p1_next:
	s_add_co_i32 s13, s12, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b4_stored
	s_add_co_i32 s13, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b4_p2_single
	global_load_b32 v48, v76, s[8:9] offset:16
	global_load_b32 v49, v76, s[8:9] offset:20
	global_load_b32 v56, v77, s[8:9] offset:16
	global_load_b32 v57, v77, s[8:9] offset:20
	global_load_b32 v64, v78, s[8:9] offset:16
	global_load_b32 v65, v78, s[8:9] offset:20
	global_load_b32 v72, v79, s[8:9] offset:16
	global_load_b32 v73, v79, s[8:9] offset:20
	s_wait_loadcnt 0x7
	v_add_f32_e32 v48, v128, v48
	global_store_b32 v76, v48, s[8:9] offset:16
	s_wait_loadcnt 0x6
	v_add_f32_e32 v49, v129, v49
	global_store_b32 v76, v49, s[8:9] offset:20
	s_wait_loadcnt 0x5
	v_add_f32_e32 v56, v152, v56
	global_store_b32 v77, v56, s[8:9] offset:16
	s_wait_loadcnt 0x4
	v_add_f32_e32 v57, v153, v57
	global_store_b32 v77, v57, s[8:9] offset:20
	s_wait_loadcnt 0x3
	v_add_f32_e32 v64, v176, v64
	global_store_b32 v78, v64, s[8:9] offset:16
	s_wait_loadcnt 0x2
	v_add_f32_e32 v65, v177, v65
	global_store_b32 v78, v65, s[8:9] offset:20
	s_wait_loadcnt 0x1
	v_add_f32_e32 v72, v200, v72
	global_store_b32 v79, v72, s[8:9] offset:16
	s_wait_loadcnt 0x0
	v_add_f32_e32 v73, v201, v73
	global_store_b32 v79, v73, s[8:9] offset:20
	s_wait_storecnt 0x0
	s_branch .Lrx_b4_p2_next
	.Lrx_b4_p2_single:
	global_load_b32 v48, v76, s[8:9] offset:16
	global_load_b32 v56, v77, s[8:9] offset:16
	global_load_b32 v64, v78, s[8:9] offset:16
	global_load_b32 v72, v79, s[8:9] offset:16
	s_wait_loadcnt 0x3
	v_add_f32_e32 v48, v48, v128
	global_store_b32 v76, v48, s[8:9] offset:16
	s_wait_loadcnt 0x2
	v_add_f32_e32 v56, v56, v152
	global_store_b32 v77, v56, s[8:9] offset:16
	s_wait_loadcnt 0x1
	v_add_f32_e32 v64, v64, v176
	global_store_b32 v78, v64, s[8:9] offset:16
	s_wait_loadcnt 0x0
	v_add_f32_e32 v72, v72, v200
	global_store_b32 v79, v72, s[8:9] offset:16
	s_wait_storecnt 0x0
	s_branch .Lrx_b4_stored
	.Lrx_b4_p2_next:
	s_add_co_i32 s13, s12, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b4_stored
	s_add_co_i32 s13, s12, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b4_p3_single
	global_load_b32 v50, v76, s[8:9] offset:24
	global_load_b32 v51, v76, s[8:9] offset:28
	global_load_b32 v58, v77, s[8:9] offset:24
	global_load_b32 v59, v77, s[8:9] offset:28
	global_load_b32 v66, v78, s[8:9] offset:24
	global_load_b32 v67, v78, s[8:9] offset:28
	global_load_b32 v74, v79, s[8:9] offset:24
	global_load_b32 v75, v79, s[8:9] offset:28
	s_wait_loadcnt 0x7
	v_add_f32_e32 v50, v130, v50
	global_store_b32 v76, v50, s[8:9] offset:24
	s_wait_loadcnt 0x6
	v_add_f32_e32 v51, v131, v51
	global_store_b32 v76, v51, s[8:9] offset:28
	s_wait_loadcnt 0x5
	v_add_f32_e32 v58, v154, v58
	global_store_b32 v77, v58, s[8:9] offset:24
	s_wait_loadcnt 0x4
	v_add_f32_e32 v59, v155, v59
	global_store_b32 v77, v59, s[8:9] offset:28
	s_wait_loadcnt 0x3
	v_add_f32_e32 v66, v178, v66
	global_store_b32 v78, v66, s[8:9] offset:24
	s_wait_loadcnt 0x2
	v_add_f32_e32 v67, v179, v67
	global_store_b32 v78, v67, s[8:9] offset:28
	s_wait_loadcnt 0x1
	v_add_f32_e32 v74, v202, v74
	global_store_b32 v79, v74, s[8:9] offset:24
	s_wait_loadcnt 0x0
	v_add_f32_e32 v75, v203, v75
	global_store_b32 v79, v75, s[8:9] offset:28
	s_wait_storecnt 0x0
	s_branch .Lrx_b4_stored
	.Lrx_b4_p3_single:
	global_load_b32 v50, v76, s[8:9] offset:24
	global_load_b32 v58, v77, s[8:9] offset:24
	global_load_b32 v66, v78, s[8:9] offset:24
	global_load_b32 v74, v79, s[8:9] offset:24
	s_wait_loadcnt 0x3
	v_add_f32_e32 v50, v50, v130
	global_store_b32 v76, v50, s[8:9] offset:24
	s_wait_loadcnt 0x2
	v_add_f32_e32 v58, v58, v154
	global_store_b32 v77, v58, s[8:9] offset:24
	s_wait_loadcnt 0x1
	v_add_f32_e32 v66, v66, v178
	global_store_b32 v78, v66, s[8:9] offset:24
	s_wait_loadcnt 0x0
	v_add_f32_e32 v74, v74, v202
	global_store_b32 v79, v74, s[8:9] offset:24
	s_wait_storecnt 0x0
	s_branch .Lrx_b4_stored
	.Lrx_b4_stored:
	s_branch .Lrx_end
	.Lrx_b5:
	s_mul_i32 s12, ttmp9, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v12, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v13, s13
	v_add_nc_u32_e32 v7, s13, v1
	s_add_co_i32 s13, s12, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v14, s13
	v_add_nc_u32_e32 v8, s13, v1
	s_add_co_i32 s13, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v15, s13
	v_add_nc_u32_e32 v9, s13, v1
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s0_done
	s_mov_b32 s15, 0x0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[186:189], v2, s[28:29]
	global_load_b128 v[190:193], v2, s[28:29] offset:16
	global_load_b128 v[194:197], v2, s[30:31]
	global_load_b128 v[198:201], v2, s[30:31] offset:16
	global_load_b128 v[202:205], v2, s[32:33]
	global_load_b128 v[206:209], v2, s[32:33] offset:16
	global_load_b128 v[210:213], v2, s[34:35]
	global_load_b128 v[214:217], v2, s[34:35] offset:16
	global_load_b128 v[218:221], v2, s[36:37]
	global_load_b128 v[222:225], v2, s[36:37] offset:16
	.Lrx_b5_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v36, v28, 0, 4
	v_bfe_u32 v37, v28, 4, 4
	v_bfe_u32 v38, v28, 8, 4
	v_bfe_u32 v39, v28, 12, 4
	v_bfe_u32 v40, v28, 16, 4
	v_bfe_u32 v41, v28, 20, 4
	v_bfe_u32 v42, v28, 24, 4
	v_bfe_u32 v43, v28, 28, 4
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_cvt_f32_ubyte0_e32 v42, v42
	v_cvt_f32_ubyte0_e32 v43, v43
	v_fma_mix_f32 v36, v16, v36, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v16, v37, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v16, v38, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v16, v39, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v46, v29, 0, 4
	v_bfe_u32 v47, v29, 4, 4
	v_bfe_u32 v48, v29, 8, 4
	v_bfe_u32 v49, v29, 12, 4
	v_bfe_u32 v50, v29, 16, 4
	v_bfe_u32 v51, v29, 20, 4
	v_bfe_u32 v52, v29, 24, 4
	v_bfe_u32 v53, v29, 28, 4
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_cvt_f32_ubyte0_e32 v52, v52
	v_cvt_f32_ubyte0_e32 v53, v53
	v_fma_mix_f32 v46, v18, v46, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v18, v47, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v18, v48, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v18, v49, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v18, v50, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v18, v51, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v18, v52, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v18, v53, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v56, v30, 0, 4
	v_bfe_u32 v57, v30, 4, 4
	v_bfe_u32 v58, v30, 8, 4
	v_bfe_u32 v59, v30, 12, 4
	v_bfe_u32 v60, v30, 16, 4
	v_bfe_u32 v61, v30, 20, 4
	v_bfe_u32 v62, v30, 24, 4
	v_bfe_u32 v63, v30, 28, 4
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f32_ubyte0_e32 v62, v62
	v_cvt_f32_ubyte0_e32 v63, v63
	v_fma_mix_f32 v56, v20, v56, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v20, v57, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v20, v58, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v20, v59, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v20, v60, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v20, v61, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v20, v62, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v20, v63, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v66, v31, 0, 4
	v_bfe_u32 v67, v31, 4, 4
	v_bfe_u32 v68, v31, 8, 4
	v_bfe_u32 v69, v31, 12, 4
	v_bfe_u32 v70, v31, 16, 4
	v_bfe_u32 v71, v31, 20, 4
	v_bfe_u32 v72, v31, 24, 4
	v_bfe_u32 v73, v31, 28, 4
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_cvt_f32_ubyte0_e32 v72, v72
	v_cvt_f32_ubyte0_e32 v73, v73
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v22, v68, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v22, v69, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v22, v70, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v22, v71, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v22, v72, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v22, v73, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v76, v32, 0, 4
	v_bfe_u32 v77, v32, 4, 4
	v_bfe_u32 v78, v32, 8, 4
	v_bfe_u32 v79, v32, 12, 4
	v_bfe_u32 v80, v32, 16, 4
	v_bfe_u32 v81, v32, 20, 4
	v_bfe_u32 v82, v32, 24, 4
	v_bfe_u32 v83, v32, 28, 4
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_cvt_f32_ubyte0_e32 v82, v82
	v_cvt_f32_ubyte0_e32 v83, v83
	v_fma_mix_f32 v76, v24, v76, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v24, v77, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v24, v78, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v24, v79, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v24, v80, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v24, v81, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v82, v24, v82, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v83, v24, v83, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v86, v33, 0, 4
	v_bfe_u32 v87, v33, 4, 4
	v_bfe_u32 v88, v33, 8, 4
	v_bfe_u32 v89, v33, 12, 4
	v_bfe_u32 v90, v33, 16, 4
	v_bfe_u32 v91, v33, 20, 4
	v_bfe_u32 v92, v33, 24, 4
	v_bfe_u32 v93, v33, 28, 4
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_cvt_f32_ubyte0_e32 v92, v92
	v_cvt_f32_ubyte0_e32 v93, v93
	v_fma_mix_f32 v86, v26, v86, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v26, v87, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v26, v88, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v26, v89, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v26, v90, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v26, v91, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v92, v26, v92, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v93, v26, v93, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v44, v36, v186 :: v_dual_mul_f32 v45, v47, v187
	v_dual_mul_f32 v54, v56, v186 :: v_dual_mul_f32 v55, v67, v187
	v_dual_mul_f32 v64, v76, v186 :: v_dual_mul_f32 v65, v87, v187
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v74, v36, v194 :: v_dual_mul_f32 v75, v47, v195
	v_dual_mul_f32 v84, v56, v194 :: v_dual_mul_f32 v85, v67, v195
	v_dual_mul_f32 v94, v76, v194 :: v_dual_mul_f32 v95, v87, v195
	v_dual_fmac_f32 v44, v37, v187 :: v_dual_fmac_f32 v45, v46, v186
	v_dual_fmac_f32 v54, v57, v187 :: v_dual_fmac_f32 v55, v66, v186
	v_dual_fmac_f32 v64, v77, v187 :: v_dual_fmac_f32 v65, v86, v186
	v_dual_fmac_f32 v74, v37, v195 :: v_dual_fmac_f32 v75, v46, v194
	v_dual_fmac_f32 v84, v57, v195 :: v_dual_fmac_f32 v85, v66, v194
	v_dual_fmac_f32 v94, v77, v195 :: v_dual_fmac_f32 v95, v86, v194
	v_dual_fmac_f32 v44, v38, v188 :: v_dual_fmac_f32 v45, v48, v188
	v_dual_fmac_f32 v54, v58, v188 :: v_dual_fmac_f32 v55, v68, v188
	v_dual_fmac_f32 v64, v78, v188 :: v_dual_fmac_f32 v65, v88, v188
	v_dual_fmac_f32 v74, v38, v196 :: v_dual_fmac_f32 v75, v48, v196
	v_dual_fmac_f32 v84, v58, v196 :: v_dual_fmac_f32 v85, v68, v196
	v_dual_fmac_f32 v94, v78, v196 :: v_dual_fmac_f32 v95, v88, v196
	v_dual_fmac_f32 v44, v39, v189 :: v_dual_fmac_f32 v45, v49, v189
	v_dual_fmac_f32 v54, v59, v189 :: v_dual_fmac_f32 v55, v69, v189
	v_dual_fmac_f32 v64, v79, v189 :: v_dual_fmac_f32 v65, v89, v189
	v_dual_fmac_f32 v74, v39, v197 :: v_dual_fmac_f32 v75, v49, v197
	v_dual_fmac_f32 v84, v59, v197 :: v_dual_fmac_f32 v85, v69, v197
	v_dual_fmac_f32 v94, v79, v197 :: v_dual_fmac_f32 v95, v89, v197
	v_dual_fmac_f32 v44, v40, v190 :: v_dual_fmac_f32 v45, v50, v190
	v_dual_fmac_f32 v54, v60, v190 :: v_dual_fmac_f32 v55, v70, v190
	v_dual_fmac_f32 v64, v80, v190 :: v_dual_fmac_f32 v65, v90, v190
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v74, v40, v198 :: v_dual_fmac_f32 v75, v50, v198
	v_dual_fmac_f32 v84, v60, v198 :: v_dual_fmac_f32 v85, v70, v198
	v_dual_fmac_f32 v94, v80, v198 :: v_dual_fmac_f32 v95, v90, v198
	v_dual_fmac_f32 v44, v41, v191 :: v_dual_fmac_f32 v45, v51, v191
	v_dual_fmac_f32 v54, v61, v191 :: v_dual_fmac_f32 v55, v71, v191
	v_dual_fmac_f32 v64, v81, v191 :: v_dual_fmac_f32 v65, v91, v191
	v_dual_fmac_f32 v74, v41, v199 :: v_dual_fmac_f32 v75, v51, v199
	v_dual_fmac_f32 v84, v61, v199 :: v_dual_fmac_f32 v85, v71, v199
	v_dual_fmac_f32 v94, v81, v199 :: v_dual_fmac_f32 v95, v91, v199
	v_dual_fmac_f32 v44, v42, v192 :: v_dual_fmac_f32 v45, v52, v192
	v_dual_fmac_f32 v54, v62, v192 :: v_dual_fmac_f32 v55, v72, v192
	v_dual_fmac_f32 v64, v82, v192 :: v_dual_fmac_f32 v65, v92, v192
	v_dual_fmac_f32 v74, v42, v200 :: v_dual_fmac_f32 v75, v52, v200
	v_dual_fmac_f32 v84, v62, v200 :: v_dual_fmac_f32 v85, v72, v200
	v_dual_fmac_f32 v94, v82, v200 :: v_dual_fmac_f32 v95, v92, v200
	v_dual_fmac_f32 v44, v43, v193 :: v_dual_fmac_f32 v45, v53, v193
	v_dual_fmac_f32 v54, v63, v193 :: v_dual_fmac_f32 v55, v73, v193
	v_dual_fmac_f32 v64, v83, v193 :: v_dual_fmac_f32 v65, v93, v193
	v_dual_fmac_f32 v74, v43, v201 :: v_dual_fmac_f32 v75, v53, v201
	v_dual_fmac_f32 v84, v63, v201 :: v_dual_fmac_f32 v85, v73, v201
	v_dual_fmac_f32 v94, v83, v201 :: v_dual_fmac_f32 v95, v93, v201
	global_load_b128 v[186:189], v3, s[28:29]
	global_load_b128 v[190:193], v3, s[28:29] offset:16
	global_load_b128 v[194:197], v3, s[30:31]
	global_load_b128 v[198:201], v3, s[30:31] offset:16
	v_dual_add_f32 v96, v44, v96 :: v_dual_add_f32 v97, v97, v45
	v_dual_add_f32 v98, v54, v98 :: v_dual_add_f32 v99, v99, v55
	v_dual_add_f32 v100, v64, v100 :: v_dual_add_f32 v101, v101, v65
	v_dual_add_f32 v114, v74, v114 :: v_dual_add_f32 v115, v115, v75
	v_dual_add_f32 v116, v84, v116 :: v_dual_add_f32 v117, v117, v85
	v_dual_add_f32 v118, v94, v118 :: v_dual_add_f32 v119, v119, v95
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v44, v36, v202 :: v_dual_mul_f32 v45, v47, v203
	v_dual_mul_f32 v54, v56, v202 :: v_dual_mul_f32 v55, v67, v203
	v_dual_mul_f32 v64, v76, v202 :: v_dual_mul_f32 v65, v87, v203
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v74, v36, v210 :: v_dual_mul_f32 v75, v47, v211
	v_dual_mul_f32 v84, v56, v210 :: v_dual_mul_f32 v85, v67, v211
	v_dual_mul_f32 v94, v76, v210 :: v_dual_mul_f32 v95, v87, v211
	v_dual_fmac_f32 v44, v37, v203 :: v_dual_fmac_f32 v45, v46, v202
	v_dual_fmac_f32 v54, v57, v203 :: v_dual_fmac_f32 v55, v66, v202
	v_dual_fmac_f32 v64, v77, v203 :: v_dual_fmac_f32 v65, v86, v202
	v_dual_fmac_f32 v74, v37, v211 :: v_dual_fmac_f32 v75, v46, v210
	v_dual_fmac_f32 v84, v57, v211 :: v_dual_fmac_f32 v85, v66, v210
	v_dual_fmac_f32 v94, v77, v211 :: v_dual_fmac_f32 v95, v86, v210
	v_dual_fmac_f32 v44, v38, v204 :: v_dual_fmac_f32 v45, v48, v204
	v_dual_fmac_f32 v54, v58, v204 :: v_dual_fmac_f32 v55, v68, v204
	v_dual_fmac_f32 v64, v78, v204 :: v_dual_fmac_f32 v65, v88, v204
	v_dual_fmac_f32 v74, v38, v212 :: v_dual_fmac_f32 v75, v48, v212
	v_dual_fmac_f32 v84, v58, v212 :: v_dual_fmac_f32 v85, v68, v212
	v_dual_fmac_f32 v94, v78, v212 :: v_dual_fmac_f32 v95, v88, v212
	v_dual_fmac_f32 v44, v39, v205 :: v_dual_fmac_f32 v45, v49, v205
	v_dual_fmac_f32 v54, v59, v205 :: v_dual_fmac_f32 v55, v69, v205
	v_dual_fmac_f32 v64, v79, v205 :: v_dual_fmac_f32 v65, v89, v205
	v_dual_fmac_f32 v74, v39, v213 :: v_dual_fmac_f32 v75, v49, v213
	v_dual_fmac_f32 v84, v59, v213 :: v_dual_fmac_f32 v85, v69, v213
	v_dual_fmac_f32 v94, v79, v213 :: v_dual_fmac_f32 v95, v89, v213
	v_dual_fmac_f32 v44, v40, v206 :: v_dual_fmac_f32 v45, v50, v206
	v_dual_fmac_f32 v54, v60, v206 :: v_dual_fmac_f32 v55, v70, v206
	v_dual_fmac_f32 v64, v80, v206 :: v_dual_fmac_f32 v65, v90, v206
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v74, v40, v214 :: v_dual_fmac_f32 v75, v50, v214
	v_dual_fmac_f32 v84, v60, v214 :: v_dual_fmac_f32 v85, v70, v214
	v_dual_fmac_f32 v94, v80, v214 :: v_dual_fmac_f32 v95, v90, v214
	v_dual_fmac_f32 v44, v41, v207 :: v_dual_fmac_f32 v45, v51, v207
	v_dual_fmac_f32 v54, v61, v207 :: v_dual_fmac_f32 v55, v71, v207
	v_dual_fmac_f32 v64, v81, v207 :: v_dual_fmac_f32 v65, v91, v207
	v_dual_fmac_f32 v74, v41, v215 :: v_dual_fmac_f32 v75, v51, v215
	v_dual_fmac_f32 v84, v61, v215 :: v_dual_fmac_f32 v85, v71, v215
	v_dual_fmac_f32 v94, v81, v215 :: v_dual_fmac_f32 v95, v91, v215
	v_dual_fmac_f32 v44, v42, v208 :: v_dual_fmac_f32 v45, v52, v208
	v_dual_fmac_f32 v54, v62, v208 :: v_dual_fmac_f32 v55, v72, v208
	v_dual_fmac_f32 v64, v82, v208 :: v_dual_fmac_f32 v65, v92, v208
	v_dual_fmac_f32 v74, v42, v216 :: v_dual_fmac_f32 v75, v52, v216
	v_dual_fmac_f32 v84, v62, v216 :: v_dual_fmac_f32 v85, v72, v216
	v_dual_fmac_f32 v94, v82, v216 :: v_dual_fmac_f32 v95, v92, v216
	v_dual_fmac_f32 v44, v43, v209 :: v_dual_fmac_f32 v45, v53, v209
	v_dual_fmac_f32 v54, v63, v209 :: v_dual_fmac_f32 v55, v73, v209
	v_dual_fmac_f32 v64, v83, v209 :: v_dual_fmac_f32 v65, v93, v209
	v_dual_fmac_f32 v74, v43, v217 :: v_dual_fmac_f32 v75, v53, v217
	v_dual_fmac_f32 v84, v63, v217 :: v_dual_fmac_f32 v85, v73, v217
	v_dual_fmac_f32 v94, v83, v217 :: v_dual_fmac_f32 v95, v93, v217
	global_load_b128 v[202:205], v3, s[32:33]
	global_load_b128 v[206:209], v3, s[32:33] offset:16
	global_load_b128 v[210:213], v3, s[34:35]
	global_load_b128 v[214:217], v3, s[34:35] offset:16
	v_dual_add_f32 v132, v44, v132 :: v_dual_add_f32 v133, v133, v45
	v_dual_add_f32 v134, v54, v134 :: v_dual_add_f32 v135, v135, v55
	v_dual_add_f32 v136, v64, v136 :: v_dual_add_f32 v137, v137, v65
	v_dual_add_f32 v150, v74, v150 :: v_dual_add_f32 v151, v151, v75
	v_dual_add_f32 v152, v84, v152 :: v_dual_add_f32 v153, v153, v85
	v_dual_add_f32 v154, v94, v154 :: v_dual_add_f32 v155, v155, v95
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v44, v36, v218 :: v_dual_mul_f32 v45, v47, v219
	v_dual_mul_f32 v54, v56, v218 :: v_dual_mul_f32 v55, v67, v219
	v_dual_mul_f32 v64, v76, v218 :: v_dual_mul_f32 v65, v87, v219
	v_dual_fmac_f32 v44, v37, v219 :: v_dual_fmac_f32 v45, v46, v218
	v_dual_fmac_f32 v54, v57, v219 :: v_dual_fmac_f32 v55, v66, v218
	v_dual_fmac_f32 v64, v77, v219 :: v_dual_fmac_f32 v65, v86, v218
	v_dual_fmac_f32 v44, v38, v220 :: v_dual_fmac_f32 v45, v48, v220
	v_dual_fmac_f32 v54, v58, v220 :: v_dual_fmac_f32 v55, v68, v220
	v_dual_fmac_f32 v64, v78, v220 :: v_dual_fmac_f32 v65, v88, v220
	v_dual_fmac_f32 v44, v39, v221 :: v_dual_fmac_f32 v45, v49, v221
	v_dual_fmac_f32 v54, v59, v221 :: v_dual_fmac_f32 v55, v69, v221
	v_dual_fmac_f32 v64, v79, v221 :: v_dual_fmac_f32 v65, v89, v221
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v44, v40, v222 :: v_dual_fmac_f32 v45, v50, v222
	v_dual_fmac_f32 v54, v60, v222 :: v_dual_fmac_f32 v55, v70, v222
	v_dual_fmac_f32 v64, v80, v222 :: v_dual_fmac_f32 v65, v90, v222
	v_dual_fmac_f32 v44, v41, v223 :: v_dual_fmac_f32 v45, v51, v223
	v_dual_fmac_f32 v54, v61, v223 :: v_dual_fmac_f32 v55, v71, v223
	v_dual_fmac_f32 v64, v81, v223 :: v_dual_fmac_f32 v65, v91, v223
	v_dual_fmac_f32 v44, v42, v224 :: v_dual_fmac_f32 v45, v52, v224
	v_dual_fmac_f32 v54, v62, v224 :: v_dual_fmac_f32 v55, v72, v224
	v_dual_fmac_f32 v64, v82, v224 :: v_dual_fmac_f32 v65, v92, v224
	v_dual_fmac_f32 v44, v43, v225 :: v_dual_fmac_f32 v45, v53, v225
	v_dual_fmac_f32 v54, v63, v225 :: v_dual_fmac_f32 v55, v73, v225
	v_dual_fmac_f32 v64, v83, v225 :: v_dual_fmac_f32 v65, v93, v225
	global_load_b128 v[218:221], v3, s[36:37]
	global_load_b128 v[222:225], v3, s[36:37] offset:16
	v_dual_add_f32 v168, v44, v168 :: v_dual_add_f32 v169, v169, v45
	v_dual_add_f32 v170, v54, v170 :: v_dual_add_f32 v171, v171, v55
	v_dual_add_f32 v172, v64, v172 :: v_dual_add_f32 v173, v173, v65
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s0
	s_wait_loadcnt 0x0
	.Lrx_b5_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[186:189], v2, s[28:29]
	global_load_b128 v[190:193], v2, s[28:29] offset:16
	global_load_b128 v[194:197], v2, s[30:31]
	global_load_b128 v[198:201], v2, s[30:31] offset:16
	global_load_b128 v[202:205], v2, s[32:33]
	global_load_b128 v[206:209], v2, s[32:33] offset:16
	global_load_b128 v[210:213], v2, s[34:35]
	global_load_b128 v[214:217], v2, s[34:35] offset:16
	global_load_b128 v[218:221], v2, s[36:37]
	global_load_b128 v[222:225], v2, s[36:37] offset:16
	.Lrx_b5_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v36, v28, 0, 4
	v_bfe_u32 v37, v28, 4, 4
	v_bfe_u32 v38, v28, 8, 4
	v_bfe_u32 v39, v28, 12, 4
	v_bfe_u32 v40, v28, 16, 4
	v_bfe_u32 v41, v28, 20, 4
	v_bfe_u32 v42, v28, 24, 4
	v_bfe_u32 v43, v28, 28, 4
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_cvt_f32_ubyte0_e32 v42, v42
	v_cvt_f32_ubyte0_e32 v43, v43
	v_fma_mix_f32 v36, v16, v36, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v16, v37, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v16, v38, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v16, v39, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v46, v29, 0, 4
	v_bfe_u32 v47, v29, 4, 4
	v_bfe_u32 v48, v29, 8, 4
	v_bfe_u32 v49, v29, 12, 4
	v_bfe_u32 v50, v29, 16, 4
	v_bfe_u32 v51, v29, 20, 4
	v_bfe_u32 v52, v29, 24, 4
	v_bfe_u32 v53, v29, 28, 4
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_cvt_f32_ubyte0_e32 v52, v52
	v_cvt_f32_ubyte0_e32 v53, v53
	v_fma_mix_f32 v46, v18, v46, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v18, v47, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v18, v48, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v18, v49, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v18, v50, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v18, v51, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v18, v52, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v18, v53, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v56, v30, 0, 4
	v_bfe_u32 v57, v30, 4, 4
	v_bfe_u32 v58, v30, 8, 4
	v_bfe_u32 v59, v30, 12, 4
	v_bfe_u32 v60, v30, 16, 4
	v_bfe_u32 v61, v30, 20, 4
	v_bfe_u32 v62, v30, 24, 4
	v_bfe_u32 v63, v30, 28, 4
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f32_ubyte0_e32 v62, v62
	v_cvt_f32_ubyte0_e32 v63, v63
	v_fma_mix_f32 v56, v20, v56, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v20, v57, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v20, v58, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v20, v59, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v20, v60, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v20, v61, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v20, v62, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v20, v63, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v66, v31, 0, 4
	v_bfe_u32 v67, v31, 4, 4
	v_bfe_u32 v68, v31, 8, 4
	v_bfe_u32 v69, v31, 12, 4
	v_bfe_u32 v70, v31, 16, 4
	v_bfe_u32 v71, v31, 20, 4
	v_bfe_u32 v72, v31, 24, 4
	v_bfe_u32 v73, v31, 28, 4
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_cvt_f32_ubyte0_e32 v72, v72
	v_cvt_f32_ubyte0_e32 v73, v73
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v22, v68, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v22, v69, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v22, v70, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v22, v71, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v22, v72, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v22, v73, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v76, v32, 0, 4
	v_bfe_u32 v77, v32, 4, 4
	v_bfe_u32 v78, v32, 8, 4
	v_bfe_u32 v79, v32, 12, 4
	v_bfe_u32 v80, v32, 16, 4
	v_bfe_u32 v81, v32, 20, 4
	v_bfe_u32 v82, v32, 24, 4
	v_bfe_u32 v83, v32, 28, 4
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_cvt_f32_ubyte0_e32 v82, v82
	v_cvt_f32_ubyte0_e32 v83, v83
	v_fma_mix_f32 v76, v24, v76, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v24, v77, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v24, v78, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v24, v79, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v24, v80, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v24, v81, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v82, v24, v82, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v83, v24, v83, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v86, v33, 0, 4
	v_bfe_u32 v87, v33, 4, 4
	v_bfe_u32 v88, v33, 8, 4
	v_bfe_u32 v89, v33, 12, 4
	v_bfe_u32 v90, v33, 16, 4
	v_bfe_u32 v91, v33, 20, 4
	v_bfe_u32 v92, v33, 24, 4
	v_bfe_u32 v93, v33, 28, 4
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_cvt_f32_ubyte0_e32 v92, v92
	v_cvt_f32_ubyte0_e32 v93, v93
	v_fma_mix_f32 v86, v26, v86, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v26, v87, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v26, v88, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v26, v89, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v26, v90, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v26, v91, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v92, v26, v92, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v93, v26, v93, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v44, v36, v186 :: v_dual_mul_f32 v45, v47, v187
	v_dual_mul_f32 v54, v56, v186 :: v_dual_mul_f32 v55, v67, v187
	v_dual_mul_f32 v64, v76, v186 :: v_dual_mul_f32 v65, v87, v187
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v74, v36, v194 :: v_dual_mul_f32 v75, v47, v195
	v_dual_mul_f32 v84, v56, v194 :: v_dual_mul_f32 v85, v67, v195
	v_dual_mul_f32 v94, v76, v194 :: v_dual_mul_f32 v95, v87, v195
	v_dual_fmac_f32 v44, v37, v187 :: v_dual_fmac_f32 v45, v46, v186
	v_dual_fmac_f32 v54, v57, v187 :: v_dual_fmac_f32 v55, v66, v186
	v_dual_fmac_f32 v64, v77, v187 :: v_dual_fmac_f32 v65, v86, v186
	v_dual_fmac_f32 v74, v37, v195 :: v_dual_fmac_f32 v75, v46, v194
	v_dual_fmac_f32 v84, v57, v195 :: v_dual_fmac_f32 v85, v66, v194
	v_dual_fmac_f32 v94, v77, v195 :: v_dual_fmac_f32 v95, v86, v194
	v_dual_fmac_f32 v44, v38, v188 :: v_dual_fmac_f32 v45, v48, v188
	v_dual_fmac_f32 v54, v58, v188 :: v_dual_fmac_f32 v55, v68, v188
	v_dual_fmac_f32 v64, v78, v188 :: v_dual_fmac_f32 v65, v88, v188
	v_dual_fmac_f32 v74, v38, v196 :: v_dual_fmac_f32 v75, v48, v196
	v_dual_fmac_f32 v84, v58, v196 :: v_dual_fmac_f32 v85, v68, v196
	v_dual_fmac_f32 v94, v78, v196 :: v_dual_fmac_f32 v95, v88, v196
	v_dual_fmac_f32 v44, v39, v189 :: v_dual_fmac_f32 v45, v49, v189
	v_dual_fmac_f32 v54, v59, v189 :: v_dual_fmac_f32 v55, v69, v189
	v_dual_fmac_f32 v64, v79, v189 :: v_dual_fmac_f32 v65, v89, v189
	v_dual_fmac_f32 v74, v39, v197 :: v_dual_fmac_f32 v75, v49, v197
	v_dual_fmac_f32 v84, v59, v197 :: v_dual_fmac_f32 v85, v69, v197
	v_dual_fmac_f32 v94, v79, v197 :: v_dual_fmac_f32 v95, v89, v197
	v_dual_fmac_f32 v44, v40, v190 :: v_dual_fmac_f32 v45, v50, v190
	v_dual_fmac_f32 v54, v60, v190 :: v_dual_fmac_f32 v55, v70, v190
	v_dual_fmac_f32 v64, v80, v190 :: v_dual_fmac_f32 v65, v90, v190
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v74, v40, v198 :: v_dual_fmac_f32 v75, v50, v198
	v_dual_fmac_f32 v84, v60, v198 :: v_dual_fmac_f32 v85, v70, v198
	v_dual_fmac_f32 v94, v80, v198 :: v_dual_fmac_f32 v95, v90, v198
	v_dual_fmac_f32 v44, v41, v191 :: v_dual_fmac_f32 v45, v51, v191
	v_dual_fmac_f32 v54, v61, v191 :: v_dual_fmac_f32 v55, v71, v191
	v_dual_fmac_f32 v64, v81, v191 :: v_dual_fmac_f32 v65, v91, v191
	v_dual_fmac_f32 v74, v41, v199 :: v_dual_fmac_f32 v75, v51, v199
	v_dual_fmac_f32 v84, v61, v199 :: v_dual_fmac_f32 v85, v71, v199
	v_dual_fmac_f32 v94, v81, v199 :: v_dual_fmac_f32 v95, v91, v199
	v_dual_fmac_f32 v44, v42, v192 :: v_dual_fmac_f32 v45, v52, v192
	v_dual_fmac_f32 v54, v62, v192 :: v_dual_fmac_f32 v55, v72, v192
	v_dual_fmac_f32 v64, v82, v192 :: v_dual_fmac_f32 v65, v92, v192
	v_dual_fmac_f32 v74, v42, v200 :: v_dual_fmac_f32 v75, v52, v200
	v_dual_fmac_f32 v84, v62, v200 :: v_dual_fmac_f32 v85, v72, v200
	v_dual_fmac_f32 v94, v82, v200 :: v_dual_fmac_f32 v95, v92, v200
	v_dual_fmac_f32 v44, v43, v193 :: v_dual_fmac_f32 v45, v53, v193
	v_dual_fmac_f32 v54, v63, v193 :: v_dual_fmac_f32 v55, v73, v193
	v_dual_fmac_f32 v64, v83, v193 :: v_dual_fmac_f32 v65, v93, v193
	v_dual_fmac_f32 v74, v43, v201 :: v_dual_fmac_f32 v75, v53, v201
	v_dual_fmac_f32 v84, v63, v201 :: v_dual_fmac_f32 v85, v73, v201
	v_dual_fmac_f32 v94, v83, v201 :: v_dual_fmac_f32 v95, v93, v201
	global_load_b128 v[186:189], v3, s[28:29]
	global_load_b128 v[190:193], v3, s[28:29] offset:16
	global_load_b128 v[194:197], v3, s[30:31]
	global_load_b128 v[198:201], v3, s[30:31] offset:16
	v_dual_add_f32 v102, v44, v102 :: v_dual_add_f32 v103, v103, v45
	v_dual_add_f32 v104, v54, v104 :: v_dual_add_f32 v105, v105, v55
	v_dual_add_f32 v106, v64, v106 :: v_dual_add_f32 v107, v107, v65
	v_dual_add_f32 v120, v74, v120 :: v_dual_add_f32 v121, v121, v75
	v_dual_add_f32 v122, v84, v122 :: v_dual_add_f32 v123, v123, v85
	v_dual_add_f32 v124, v94, v124 :: v_dual_add_f32 v125, v125, v95
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v44, v36, v202 :: v_dual_mul_f32 v45, v47, v203
	v_dual_mul_f32 v54, v56, v202 :: v_dual_mul_f32 v55, v67, v203
	v_dual_mul_f32 v64, v76, v202 :: v_dual_mul_f32 v65, v87, v203
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v74, v36, v210 :: v_dual_mul_f32 v75, v47, v211
	v_dual_mul_f32 v84, v56, v210 :: v_dual_mul_f32 v85, v67, v211
	v_dual_mul_f32 v94, v76, v210 :: v_dual_mul_f32 v95, v87, v211
	v_dual_fmac_f32 v44, v37, v203 :: v_dual_fmac_f32 v45, v46, v202
	v_dual_fmac_f32 v54, v57, v203 :: v_dual_fmac_f32 v55, v66, v202
	v_dual_fmac_f32 v64, v77, v203 :: v_dual_fmac_f32 v65, v86, v202
	v_dual_fmac_f32 v74, v37, v211 :: v_dual_fmac_f32 v75, v46, v210
	v_dual_fmac_f32 v84, v57, v211 :: v_dual_fmac_f32 v85, v66, v210
	v_dual_fmac_f32 v94, v77, v211 :: v_dual_fmac_f32 v95, v86, v210
	v_dual_fmac_f32 v44, v38, v204 :: v_dual_fmac_f32 v45, v48, v204
	v_dual_fmac_f32 v54, v58, v204 :: v_dual_fmac_f32 v55, v68, v204
	v_dual_fmac_f32 v64, v78, v204 :: v_dual_fmac_f32 v65, v88, v204
	v_dual_fmac_f32 v74, v38, v212 :: v_dual_fmac_f32 v75, v48, v212
	v_dual_fmac_f32 v84, v58, v212 :: v_dual_fmac_f32 v85, v68, v212
	v_dual_fmac_f32 v94, v78, v212 :: v_dual_fmac_f32 v95, v88, v212
	v_dual_fmac_f32 v44, v39, v205 :: v_dual_fmac_f32 v45, v49, v205
	v_dual_fmac_f32 v54, v59, v205 :: v_dual_fmac_f32 v55, v69, v205
	v_dual_fmac_f32 v64, v79, v205 :: v_dual_fmac_f32 v65, v89, v205
	v_dual_fmac_f32 v74, v39, v213 :: v_dual_fmac_f32 v75, v49, v213
	v_dual_fmac_f32 v84, v59, v213 :: v_dual_fmac_f32 v85, v69, v213
	v_dual_fmac_f32 v94, v79, v213 :: v_dual_fmac_f32 v95, v89, v213
	v_dual_fmac_f32 v44, v40, v206 :: v_dual_fmac_f32 v45, v50, v206
	v_dual_fmac_f32 v54, v60, v206 :: v_dual_fmac_f32 v55, v70, v206
	v_dual_fmac_f32 v64, v80, v206 :: v_dual_fmac_f32 v65, v90, v206
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v74, v40, v214 :: v_dual_fmac_f32 v75, v50, v214
	v_dual_fmac_f32 v84, v60, v214 :: v_dual_fmac_f32 v85, v70, v214
	v_dual_fmac_f32 v94, v80, v214 :: v_dual_fmac_f32 v95, v90, v214
	v_dual_fmac_f32 v44, v41, v207 :: v_dual_fmac_f32 v45, v51, v207
	v_dual_fmac_f32 v54, v61, v207 :: v_dual_fmac_f32 v55, v71, v207
	v_dual_fmac_f32 v64, v81, v207 :: v_dual_fmac_f32 v65, v91, v207
	v_dual_fmac_f32 v74, v41, v215 :: v_dual_fmac_f32 v75, v51, v215
	v_dual_fmac_f32 v84, v61, v215 :: v_dual_fmac_f32 v85, v71, v215
	v_dual_fmac_f32 v94, v81, v215 :: v_dual_fmac_f32 v95, v91, v215
	v_dual_fmac_f32 v44, v42, v208 :: v_dual_fmac_f32 v45, v52, v208
	v_dual_fmac_f32 v54, v62, v208 :: v_dual_fmac_f32 v55, v72, v208
	v_dual_fmac_f32 v64, v82, v208 :: v_dual_fmac_f32 v65, v92, v208
	v_dual_fmac_f32 v74, v42, v216 :: v_dual_fmac_f32 v75, v52, v216
	v_dual_fmac_f32 v84, v62, v216 :: v_dual_fmac_f32 v85, v72, v216
	v_dual_fmac_f32 v94, v82, v216 :: v_dual_fmac_f32 v95, v92, v216
	v_dual_fmac_f32 v44, v43, v209 :: v_dual_fmac_f32 v45, v53, v209
	v_dual_fmac_f32 v54, v63, v209 :: v_dual_fmac_f32 v55, v73, v209
	v_dual_fmac_f32 v64, v83, v209 :: v_dual_fmac_f32 v65, v93, v209
	v_dual_fmac_f32 v74, v43, v217 :: v_dual_fmac_f32 v75, v53, v217
	v_dual_fmac_f32 v84, v63, v217 :: v_dual_fmac_f32 v85, v73, v217
	v_dual_fmac_f32 v94, v83, v217 :: v_dual_fmac_f32 v95, v93, v217
	global_load_b128 v[202:205], v3, s[32:33]
	global_load_b128 v[206:209], v3, s[32:33] offset:16
	global_load_b128 v[210:213], v3, s[34:35]
	global_load_b128 v[214:217], v3, s[34:35] offset:16
	v_dual_add_f32 v138, v44, v138 :: v_dual_add_f32 v139, v139, v45
	v_dual_add_f32 v140, v54, v140 :: v_dual_add_f32 v141, v141, v55
	v_dual_add_f32 v142, v64, v142 :: v_dual_add_f32 v143, v143, v65
	v_dual_add_f32 v156, v74, v156 :: v_dual_add_f32 v157, v157, v75
	v_dual_add_f32 v158, v84, v158 :: v_dual_add_f32 v159, v159, v85
	v_dual_add_f32 v160, v94, v160 :: v_dual_add_f32 v161, v161, v95
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v44, v36, v218 :: v_dual_mul_f32 v45, v47, v219
	v_dual_mul_f32 v54, v56, v218 :: v_dual_mul_f32 v55, v67, v219
	v_dual_mul_f32 v64, v76, v218 :: v_dual_mul_f32 v65, v87, v219
	v_dual_fmac_f32 v44, v37, v219 :: v_dual_fmac_f32 v45, v46, v218
	v_dual_fmac_f32 v54, v57, v219 :: v_dual_fmac_f32 v55, v66, v218
	v_dual_fmac_f32 v64, v77, v219 :: v_dual_fmac_f32 v65, v86, v218
	v_dual_fmac_f32 v44, v38, v220 :: v_dual_fmac_f32 v45, v48, v220
	v_dual_fmac_f32 v54, v58, v220 :: v_dual_fmac_f32 v55, v68, v220
	v_dual_fmac_f32 v64, v78, v220 :: v_dual_fmac_f32 v65, v88, v220
	v_dual_fmac_f32 v44, v39, v221 :: v_dual_fmac_f32 v45, v49, v221
	v_dual_fmac_f32 v54, v59, v221 :: v_dual_fmac_f32 v55, v69, v221
	v_dual_fmac_f32 v64, v79, v221 :: v_dual_fmac_f32 v65, v89, v221
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v44, v40, v222 :: v_dual_fmac_f32 v45, v50, v222
	v_dual_fmac_f32 v54, v60, v222 :: v_dual_fmac_f32 v55, v70, v222
	v_dual_fmac_f32 v64, v80, v222 :: v_dual_fmac_f32 v65, v90, v222
	v_dual_fmac_f32 v44, v41, v223 :: v_dual_fmac_f32 v45, v51, v223
	v_dual_fmac_f32 v54, v61, v223 :: v_dual_fmac_f32 v55, v71, v223
	v_dual_fmac_f32 v64, v81, v223 :: v_dual_fmac_f32 v65, v91, v223
	v_dual_fmac_f32 v44, v42, v224 :: v_dual_fmac_f32 v45, v52, v224
	v_dual_fmac_f32 v54, v62, v224 :: v_dual_fmac_f32 v55, v72, v224
	v_dual_fmac_f32 v64, v82, v224 :: v_dual_fmac_f32 v65, v92, v224
	v_dual_fmac_f32 v44, v43, v225 :: v_dual_fmac_f32 v45, v53, v225
	v_dual_fmac_f32 v54, v63, v225 :: v_dual_fmac_f32 v55, v73, v225
	v_dual_fmac_f32 v64, v83, v225 :: v_dual_fmac_f32 v65, v93, v225
	global_load_b128 v[218:221], v3, s[36:37]
	global_load_b128 v[222:225], v3, s[36:37] offset:16
	v_dual_add_f32 v174, v44, v174 :: v_dual_add_f32 v175, v175, v45
	v_dual_add_f32 v176, v54, v176 :: v_dual_add_f32 v177, v177, v55
	v_dual_add_f32 v178, v64, v178 :: v_dual_add_f32 v179, v179, v65
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s1
	s_wait_loadcnt 0x0
	.Lrx_b5_s1_done:
	v_dual_add_f32 v96, v96, v102 :: v_dual_add_f32 v97, v97, v103
	v_dual_add_f32 v98, v98, v104 :: v_dual_add_f32 v99, v99, v105
	v_dual_add_f32 v100, v100, v106 :: v_dual_add_f32 v101, v101, v107
	v_dual_add_f32 v114, v114, v120 :: v_dual_add_f32 v115, v115, v121
	v_dual_add_f32 v116, v116, v122 :: v_dual_add_f32 v117, v117, v123
	v_dual_add_f32 v118, v118, v124 :: v_dual_add_f32 v119, v119, v125
	v_dual_add_f32 v132, v132, v138 :: v_dual_add_f32 v133, v133, v139
	v_dual_add_f32 v134, v134, v140 :: v_dual_add_f32 v135, v135, v141
	v_dual_add_f32 v136, v136, v142 :: v_dual_add_f32 v137, v137, v143
	v_dual_add_f32 v150, v150, v156 :: v_dual_add_f32 v151, v151, v157
	v_dual_add_f32 v152, v152, v158 :: v_dual_add_f32 v153, v153, v159
	v_dual_add_f32 v154, v154, v160 :: v_dual_add_f32 v155, v155, v161
	v_dual_add_f32 v168, v168, v174 :: v_dual_add_f32 v169, v169, v175
	v_dual_add_f32 v170, v170, v176 :: v_dual_add_f32 v171, v171, v177
	v_dual_add_f32 v172, v172, v178 :: v_dual_add_f32 v173, v173, v179
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[186:189], v2, s[28:29]
	global_load_b128 v[190:193], v2, s[28:29] offset:16
	global_load_b128 v[194:197], v2, s[30:31]
	global_load_b128 v[198:201], v2, s[30:31] offset:16
	global_load_b128 v[202:205], v2, s[32:33]
	global_load_b128 v[206:209], v2, s[32:33] offset:16
	global_load_b128 v[210:213], v2, s[34:35]
	global_load_b128 v[214:217], v2, s[34:35] offset:16
	global_load_b128 v[218:221], v2, s[36:37]
	global_load_b128 v[222:225], v2, s[36:37] offset:16
	.Lrx_b5_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v36, v28, 0, 4
	v_bfe_u32 v37, v28, 4, 4
	v_bfe_u32 v38, v28, 8, 4
	v_bfe_u32 v39, v28, 12, 4
	v_bfe_u32 v40, v28, 16, 4
	v_bfe_u32 v41, v28, 20, 4
	v_bfe_u32 v42, v28, 24, 4
	v_bfe_u32 v43, v28, 28, 4
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_cvt_f32_ubyte0_e32 v42, v42
	v_cvt_f32_ubyte0_e32 v43, v43
	v_fma_mix_f32 v36, v16, v36, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v16, v37, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v16, v38, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v16, v39, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v46, v29, 0, 4
	v_bfe_u32 v47, v29, 4, 4
	v_bfe_u32 v48, v29, 8, 4
	v_bfe_u32 v49, v29, 12, 4
	v_bfe_u32 v50, v29, 16, 4
	v_bfe_u32 v51, v29, 20, 4
	v_bfe_u32 v52, v29, 24, 4
	v_bfe_u32 v53, v29, 28, 4
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_cvt_f32_ubyte0_e32 v52, v52
	v_cvt_f32_ubyte0_e32 v53, v53
	v_fma_mix_f32 v46, v18, v46, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v18, v47, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v18, v48, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v18, v49, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v18, v50, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v18, v51, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v18, v52, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v18, v53, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v56, v30, 0, 4
	v_bfe_u32 v57, v30, 4, 4
	v_bfe_u32 v58, v30, 8, 4
	v_bfe_u32 v59, v30, 12, 4
	v_bfe_u32 v60, v30, 16, 4
	v_bfe_u32 v61, v30, 20, 4
	v_bfe_u32 v62, v30, 24, 4
	v_bfe_u32 v63, v30, 28, 4
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f32_ubyte0_e32 v62, v62
	v_cvt_f32_ubyte0_e32 v63, v63
	v_fma_mix_f32 v56, v20, v56, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v20, v57, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v20, v58, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v20, v59, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v20, v60, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v20, v61, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v20, v62, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v20, v63, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v66, v31, 0, 4
	v_bfe_u32 v67, v31, 4, 4
	v_bfe_u32 v68, v31, 8, 4
	v_bfe_u32 v69, v31, 12, 4
	v_bfe_u32 v70, v31, 16, 4
	v_bfe_u32 v71, v31, 20, 4
	v_bfe_u32 v72, v31, 24, 4
	v_bfe_u32 v73, v31, 28, 4
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_cvt_f32_ubyte0_e32 v72, v72
	v_cvt_f32_ubyte0_e32 v73, v73
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v22, v68, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v22, v69, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v22, v70, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v22, v71, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v22, v72, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v22, v73, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v76, v32, 0, 4
	v_bfe_u32 v77, v32, 4, 4
	v_bfe_u32 v78, v32, 8, 4
	v_bfe_u32 v79, v32, 12, 4
	v_bfe_u32 v80, v32, 16, 4
	v_bfe_u32 v81, v32, 20, 4
	v_bfe_u32 v82, v32, 24, 4
	v_bfe_u32 v83, v32, 28, 4
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_cvt_f32_ubyte0_e32 v82, v82
	v_cvt_f32_ubyte0_e32 v83, v83
	v_fma_mix_f32 v76, v24, v76, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v24, v77, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v24, v78, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v24, v79, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v24, v80, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v24, v81, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v82, v24, v82, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v83, v24, v83, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v86, v33, 0, 4
	v_bfe_u32 v87, v33, 4, 4
	v_bfe_u32 v88, v33, 8, 4
	v_bfe_u32 v89, v33, 12, 4
	v_bfe_u32 v90, v33, 16, 4
	v_bfe_u32 v91, v33, 20, 4
	v_bfe_u32 v92, v33, 24, 4
	v_bfe_u32 v93, v33, 28, 4
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_cvt_f32_ubyte0_e32 v92, v92
	v_cvt_f32_ubyte0_e32 v93, v93
	v_fma_mix_f32 v86, v26, v86, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v26, v87, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v26, v88, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v26, v89, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v26, v90, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v26, v91, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v92, v26, v92, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v93, v26, v93, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v44, v36, v186 :: v_dual_mul_f32 v45, v47, v187
	v_dual_mul_f32 v54, v56, v186 :: v_dual_mul_f32 v55, v67, v187
	v_dual_mul_f32 v64, v76, v186 :: v_dual_mul_f32 v65, v87, v187
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v74, v36, v194 :: v_dual_mul_f32 v75, v47, v195
	v_dual_mul_f32 v84, v56, v194 :: v_dual_mul_f32 v85, v67, v195
	v_dual_mul_f32 v94, v76, v194 :: v_dual_mul_f32 v95, v87, v195
	v_dual_fmac_f32 v44, v37, v187 :: v_dual_fmac_f32 v45, v46, v186
	v_dual_fmac_f32 v54, v57, v187 :: v_dual_fmac_f32 v55, v66, v186
	v_dual_fmac_f32 v64, v77, v187 :: v_dual_fmac_f32 v65, v86, v186
	v_dual_fmac_f32 v74, v37, v195 :: v_dual_fmac_f32 v75, v46, v194
	v_dual_fmac_f32 v84, v57, v195 :: v_dual_fmac_f32 v85, v66, v194
	v_dual_fmac_f32 v94, v77, v195 :: v_dual_fmac_f32 v95, v86, v194
	v_dual_fmac_f32 v44, v38, v188 :: v_dual_fmac_f32 v45, v48, v188
	v_dual_fmac_f32 v54, v58, v188 :: v_dual_fmac_f32 v55, v68, v188
	v_dual_fmac_f32 v64, v78, v188 :: v_dual_fmac_f32 v65, v88, v188
	v_dual_fmac_f32 v74, v38, v196 :: v_dual_fmac_f32 v75, v48, v196
	v_dual_fmac_f32 v84, v58, v196 :: v_dual_fmac_f32 v85, v68, v196
	v_dual_fmac_f32 v94, v78, v196 :: v_dual_fmac_f32 v95, v88, v196
	v_dual_fmac_f32 v44, v39, v189 :: v_dual_fmac_f32 v45, v49, v189
	v_dual_fmac_f32 v54, v59, v189 :: v_dual_fmac_f32 v55, v69, v189
	v_dual_fmac_f32 v64, v79, v189 :: v_dual_fmac_f32 v65, v89, v189
	v_dual_fmac_f32 v74, v39, v197 :: v_dual_fmac_f32 v75, v49, v197
	v_dual_fmac_f32 v84, v59, v197 :: v_dual_fmac_f32 v85, v69, v197
	v_dual_fmac_f32 v94, v79, v197 :: v_dual_fmac_f32 v95, v89, v197
	v_dual_fmac_f32 v44, v40, v190 :: v_dual_fmac_f32 v45, v50, v190
	v_dual_fmac_f32 v54, v60, v190 :: v_dual_fmac_f32 v55, v70, v190
	v_dual_fmac_f32 v64, v80, v190 :: v_dual_fmac_f32 v65, v90, v190
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v74, v40, v198 :: v_dual_fmac_f32 v75, v50, v198
	v_dual_fmac_f32 v84, v60, v198 :: v_dual_fmac_f32 v85, v70, v198
	v_dual_fmac_f32 v94, v80, v198 :: v_dual_fmac_f32 v95, v90, v198
	v_dual_fmac_f32 v44, v41, v191 :: v_dual_fmac_f32 v45, v51, v191
	v_dual_fmac_f32 v54, v61, v191 :: v_dual_fmac_f32 v55, v71, v191
	v_dual_fmac_f32 v64, v81, v191 :: v_dual_fmac_f32 v65, v91, v191
	v_dual_fmac_f32 v74, v41, v199 :: v_dual_fmac_f32 v75, v51, v199
	v_dual_fmac_f32 v84, v61, v199 :: v_dual_fmac_f32 v85, v71, v199
	v_dual_fmac_f32 v94, v81, v199 :: v_dual_fmac_f32 v95, v91, v199
	v_dual_fmac_f32 v44, v42, v192 :: v_dual_fmac_f32 v45, v52, v192
	v_dual_fmac_f32 v54, v62, v192 :: v_dual_fmac_f32 v55, v72, v192
	v_dual_fmac_f32 v64, v82, v192 :: v_dual_fmac_f32 v65, v92, v192
	v_dual_fmac_f32 v74, v42, v200 :: v_dual_fmac_f32 v75, v52, v200
	v_dual_fmac_f32 v84, v62, v200 :: v_dual_fmac_f32 v85, v72, v200
	v_dual_fmac_f32 v94, v82, v200 :: v_dual_fmac_f32 v95, v92, v200
	v_dual_fmac_f32 v44, v43, v193 :: v_dual_fmac_f32 v45, v53, v193
	v_dual_fmac_f32 v54, v63, v193 :: v_dual_fmac_f32 v55, v73, v193
	v_dual_fmac_f32 v64, v83, v193 :: v_dual_fmac_f32 v65, v93, v193
	v_dual_fmac_f32 v74, v43, v201 :: v_dual_fmac_f32 v75, v53, v201
	v_dual_fmac_f32 v84, v63, v201 :: v_dual_fmac_f32 v85, v73, v201
	v_dual_fmac_f32 v94, v83, v201 :: v_dual_fmac_f32 v95, v93, v201
	global_load_b128 v[186:189], v3, s[28:29]
	global_load_b128 v[190:193], v3, s[28:29] offset:16
	global_load_b128 v[194:197], v3, s[30:31]
	global_load_b128 v[198:201], v3, s[30:31] offset:16
	v_dual_add_f32 v102, v44, v102 :: v_dual_add_f32 v103, v103, v45
	v_dual_add_f32 v104, v54, v104 :: v_dual_add_f32 v105, v105, v55
	v_dual_add_f32 v106, v64, v106 :: v_dual_add_f32 v107, v107, v65
	v_dual_add_f32 v120, v74, v120 :: v_dual_add_f32 v121, v121, v75
	v_dual_add_f32 v122, v84, v122 :: v_dual_add_f32 v123, v123, v85
	v_dual_add_f32 v124, v94, v124 :: v_dual_add_f32 v125, v125, v95
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v44, v36, v202 :: v_dual_mul_f32 v45, v47, v203
	v_dual_mul_f32 v54, v56, v202 :: v_dual_mul_f32 v55, v67, v203
	v_dual_mul_f32 v64, v76, v202 :: v_dual_mul_f32 v65, v87, v203
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v74, v36, v210 :: v_dual_mul_f32 v75, v47, v211
	v_dual_mul_f32 v84, v56, v210 :: v_dual_mul_f32 v85, v67, v211
	v_dual_mul_f32 v94, v76, v210 :: v_dual_mul_f32 v95, v87, v211
	v_dual_fmac_f32 v44, v37, v203 :: v_dual_fmac_f32 v45, v46, v202
	v_dual_fmac_f32 v54, v57, v203 :: v_dual_fmac_f32 v55, v66, v202
	v_dual_fmac_f32 v64, v77, v203 :: v_dual_fmac_f32 v65, v86, v202
	v_dual_fmac_f32 v74, v37, v211 :: v_dual_fmac_f32 v75, v46, v210
	v_dual_fmac_f32 v84, v57, v211 :: v_dual_fmac_f32 v85, v66, v210
	v_dual_fmac_f32 v94, v77, v211 :: v_dual_fmac_f32 v95, v86, v210
	v_dual_fmac_f32 v44, v38, v204 :: v_dual_fmac_f32 v45, v48, v204
	v_dual_fmac_f32 v54, v58, v204 :: v_dual_fmac_f32 v55, v68, v204
	v_dual_fmac_f32 v64, v78, v204 :: v_dual_fmac_f32 v65, v88, v204
	v_dual_fmac_f32 v74, v38, v212 :: v_dual_fmac_f32 v75, v48, v212
	v_dual_fmac_f32 v84, v58, v212 :: v_dual_fmac_f32 v85, v68, v212
	v_dual_fmac_f32 v94, v78, v212 :: v_dual_fmac_f32 v95, v88, v212
	v_dual_fmac_f32 v44, v39, v205 :: v_dual_fmac_f32 v45, v49, v205
	v_dual_fmac_f32 v54, v59, v205 :: v_dual_fmac_f32 v55, v69, v205
	v_dual_fmac_f32 v64, v79, v205 :: v_dual_fmac_f32 v65, v89, v205
	v_dual_fmac_f32 v74, v39, v213 :: v_dual_fmac_f32 v75, v49, v213
	v_dual_fmac_f32 v84, v59, v213 :: v_dual_fmac_f32 v85, v69, v213
	v_dual_fmac_f32 v94, v79, v213 :: v_dual_fmac_f32 v95, v89, v213
	v_dual_fmac_f32 v44, v40, v206 :: v_dual_fmac_f32 v45, v50, v206
	v_dual_fmac_f32 v54, v60, v206 :: v_dual_fmac_f32 v55, v70, v206
	v_dual_fmac_f32 v64, v80, v206 :: v_dual_fmac_f32 v65, v90, v206
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v74, v40, v214 :: v_dual_fmac_f32 v75, v50, v214
	v_dual_fmac_f32 v84, v60, v214 :: v_dual_fmac_f32 v85, v70, v214
	v_dual_fmac_f32 v94, v80, v214 :: v_dual_fmac_f32 v95, v90, v214
	v_dual_fmac_f32 v44, v41, v207 :: v_dual_fmac_f32 v45, v51, v207
	v_dual_fmac_f32 v54, v61, v207 :: v_dual_fmac_f32 v55, v71, v207
	v_dual_fmac_f32 v64, v81, v207 :: v_dual_fmac_f32 v65, v91, v207
	v_dual_fmac_f32 v74, v41, v215 :: v_dual_fmac_f32 v75, v51, v215
	v_dual_fmac_f32 v84, v61, v215 :: v_dual_fmac_f32 v85, v71, v215
	v_dual_fmac_f32 v94, v81, v215 :: v_dual_fmac_f32 v95, v91, v215
	v_dual_fmac_f32 v44, v42, v208 :: v_dual_fmac_f32 v45, v52, v208
	v_dual_fmac_f32 v54, v62, v208 :: v_dual_fmac_f32 v55, v72, v208
	v_dual_fmac_f32 v64, v82, v208 :: v_dual_fmac_f32 v65, v92, v208
	v_dual_fmac_f32 v74, v42, v216 :: v_dual_fmac_f32 v75, v52, v216
	v_dual_fmac_f32 v84, v62, v216 :: v_dual_fmac_f32 v85, v72, v216
	v_dual_fmac_f32 v94, v82, v216 :: v_dual_fmac_f32 v95, v92, v216
	v_dual_fmac_f32 v44, v43, v209 :: v_dual_fmac_f32 v45, v53, v209
	v_dual_fmac_f32 v54, v63, v209 :: v_dual_fmac_f32 v55, v73, v209
	v_dual_fmac_f32 v64, v83, v209 :: v_dual_fmac_f32 v65, v93, v209
	v_dual_fmac_f32 v74, v43, v217 :: v_dual_fmac_f32 v75, v53, v217
	v_dual_fmac_f32 v84, v63, v217 :: v_dual_fmac_f32 v85, v73, v217
	v_dual_fmac_f32 v94, v83, v217 :: v_dual_fmac_f32 v95, v93, v217
	global_load_b128 v[202:205], v3, s[32:33]
	global_load_b128 v[206:209], v3, s[32:33] offset:16
	global_load_b128 v[210:213], v3, s[34:35]
	global_load_b128 v[214:217], v3, s[34:35] offset:16
	v_dual_add_f32 v138, v44, v138 :: v_dual_add_f32 v139, v139, v45
	v_dual_add_f32 v140, v54, v140 :: v_dual_add_f32 v141, v141, v55
	v_dual_add_f32 v142, v64, v142 :: v_dual_add_f32 v143, v143, v65
	v_dual_add_f32 v156, v74, v156 :: v_dual_add_f32 v157, v157, v75
	v_dual_add_f32 v158, v84, v158 :: v_dual_add_f32 v159, v159, v85
	v_dual_add_f32 v160, v94, v160 :: v_dual_add_f32 v161, v161, v95
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v44, v36, v218 :: v_dual_mul_f32 v45, v47, v219
	v_dual_mul_f32 v54, v56, v218 :: v_dual_mul_f32 v55, v67, v219
	v_dual_mul_f32 v64, v76, v218 :: v_dual_mul_f32 v65, v87, v219
	v_dual_fmac_f32 v44, v37, v219 :: v_dual_fmac_f32 v45, v46, v218
	v_dual_fmac_f32 v54, v57, v219 :: v_dual_fmac_f32 v55, v66, v218
	v_dual_fmac_f32 v64, v77, v219 :: v_dual_fmac_f32 v65, v86, v218
	v_dual_fmac_f32 v44, v38, v220 :: v_dual_fmac_f32 v45, v48, v220
	v_dual_fmac_f32 v54, v58, v220 :: v_dual_fmac_f32 v55, v68, v220
	v_dual_fmac_f32 v64, v78, v220 :: v_dual_fmac_f32 v65, v88, v220
	v_dual_fmac_f32 v44, v39, v221 :: v_dual_fmac_f32 v45, v49, v221
	v_dual_fmac_f32 v54, v59, v221 :: v_dual_fmac_f32 v55, v69, v221
	v_dual_fmac_f32 v64, v79, v221 :: v_dual_fmac_f32 v65, v89, v221
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v44, v40, v222 :: v_dual_fmac_f32 v45, v50, v222
	v_dual_fmac_f32 v54, v60, v222 :: v_dual_fmac_f32 v55, v70, v222
	v_dual_fmac_f32 v64, v80, v222 :: v_dual_fmac_f32 v65, v90, v222
	v_dual_fmac_f32 v44, v41, v223 :: v_dual_fmac_f32 v45, v51, v223
	v_dual_fmac_f32 v54, v61, v223 :: v_dual_fmac_f32 v55, v71, v223
	v_dual_fmac_f32 v64, v81, v223 :: v_dual_fmac_f32 v65, v91, v223
	v_dual_fmac_f32 v44, v42, v224 :: v_dual_fmac_f32 v45, v52, v224
	v_dual_fmac_f32 v54, v62, v224 :: v_dual_fmac_f32 v55, v72, v224
	v_dual_fmac_f32 v64, v82, v224 :: v_dual_fmac_f32 v65, v92, v224
	v_dual_fmac_f32 v44, v43, v225 :: v_dual_fmac_f32 v45, v53, v225
	v_dual_fmac_f32 v54, v63, v225 :: v_dual_fmac_f32 v55, v73, v225
	v_dual_fmac_f32 v64, v83, v225 :: v_dual_fmac_f32 v65, v93, v225
	global_load_b128 v[218:221], v3, s[36:37]
	global_load_b128 v[222:225], v3, s[36:37] offset:16
	v_dual_add_f32 v174, v44, v174 :: v_dual_add_f32 v175, v175, v45
	v_dual_add_f32 v176, v54, v176 :: v_dual_add_f32 v177, v177, v55
	v_dual_add_f32 v178, v64, v178 :: v_dual_add_f32 v179, v179, v65
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s2
	s_wait_loadcnt 0x0
	.Lrx_b5_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[186:189], v2, s[28:29]
	global_load_b128 v[190:193], v2, s[28:29] offset:16
	global_load_b128 v[194:197], v2, s[30:31]
	global_load_b128 v[198:201], v2, s[30:31] offset:16
	global_load_b128 v[202:205], v2, s[32:33]
	global_load_b128 v[206:209], v2, s[32:33] offset:16
	global_load_b128 v[210:213], v2, s[34:35]
	global_load_b128 v[214:217], v2, s[34:35] offset:16
	global_load_b128 v[218:221], v2, s[36:37]
	global_load_b128 v[222:225], v2, s[36:37] offset:16
	.Lrx_b5_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v36, v28, 0, 4
	v_bfe_u32 v37, v28, 4, 4
	v_bfe_u32 v38, v28, 8, 4
	v_bfe_u32 v39, v28, 12, 4
	v_bfe_u32 v40, v28, 16, 4
	v_bfe_u32 v41, v28, 20, 4
	v_bfe_u32 v42, v28, 24, 4
	v_bfe_u32 v43, v28, 28, 4
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_cvt_f32_ubyte0_e32 v42, v42
	v_cvt_f32_ubyte0_e32 v43, v43
	v_fma_mix_f32 v36, v16, v36, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v16, v37, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v16, v38, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v16, v39, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v46, v29, 0, 4
	v_bfe_u32 v47, v29, 4, 4
	v_bfe_u32 v48, v29, 8, 4
	v_bfe_u32 v49, v29, 12, 4
	v_bfe_u32 v50, v29, 16, 4
	v_bfe_u32 v51, v29, 20, 4
	v_bfe_u32 v52, v29, 24, 4
	v_bfe_u32 v53, v29, 28, 4
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_cvt_f32_ubyte0_e32 v52, v52
	v_cvt_f32_ubyte0_e32 v53, v53
	v_fma_mix_f32 v46, v18, v46, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v18, v47, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v18, v48, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v18, v49, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v18, v50, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v18, v51, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v18, v52, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v18, v53, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v56, v30, 0, 4
	v_bfe_u32 v57, v30, 4, 4
	v_bfe_u32 v58, v30, 8, 4
	v_bfe_u32 v59, v30, 12, 4
	v_bfe_u32 v60, v30, 16, 4
	v_bfe_u32 v61, v30, 20, 4
	v_bfe_u32 v62, v30, 24, 4
	v_bfe_u32 v63, v30, 28, 4
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f32_ubyte0_e32 v62, v62
	v_cvt_f32_ubyte0_e32 v63, v63
	v_fma_mix_f32 v56, v20, v56, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v20, v57, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v20, v58, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v20, v59, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v20, v60, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v20, v61, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v20, v62, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v20, v63, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v66, v31, 0, 4
	v_bfe_u32 v67, v31, 4, 4
	v_bfe_u32 v68, v31, 8, 4
	v_bfe_u32 v69, v31, 12, 4
	v_bfe_u32 v70, v31, 16, 4
	v_bfe_u32 v71, v31, 20, 4
	v_bfe_u32 v72, v31, 24, 4
	v_bfe_u32 v73, v31, 28, 4
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_cvt_f32_ubyte0_e32 v72, v72
	v_cvt_f32_ubyte0_e32 v73, v73
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v22, v68, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v22, v69, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v22, v70, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v22, v71, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v22, v72, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v22, v73, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v76, v32, 0, 4
	v_bfe_u32 v77, v32, 4, 4
	v_bfe_u32 v78, v32, 8, 4
	v_bfe_u32 v79, v32, 12, 4
	v_bfe_u32 v80, v32, 16, 4
	v_bfe_u32 v81, v32, 20, 4
	v_bfe_u32 v82, v32, 24, 4
	v_bfe_u32 v83, v32, 28, 4
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_cvt_f32_ubyte0_e32 v82, v82
	v_cvt_f32_ubyte0_e32 v83, v83
	v_fma_mix_f32 v76, v24, v76, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v24, v77, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v24, v78, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v24, v79, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v24, v80, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v24, v81, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v82, v24, v82, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v83, v24, v83, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xa
	v_bfe_u32 v86, v33, 0, 4
	v_bfe_u32 v87, v33, 4, 4
	v_bfe_u32 v88, v33, 8, 4
	v_bfe_u32 v89, v33, 12, 4
	v_bfe_u32 v90, v33, 16, 4
	v_bfe_u32 v91, v33, 20, 4
	v_bfe_u32 v92, v33, 24, 4
	v_bfe_u32 v93, v33, 28, 4
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_cvt_f32_ubyte0_e32 v92, v92
	v_cvt_f32_ubyte0_e32 v93, v93
	v_fma_mix_f32 v86, v26, v86, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v26, v87, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v26, v88, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v26, v89, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v26, v90, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v26, v91, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v92, v26, v92, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v93, v26, v93, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v44, v36, v186 :: v_dual_mul_f32 v45, v47, v187
	v_dual_mul_f32 v54, v56, v186 :: v_dual_mul_f32 v55, v67, v187
	v_dual_mul_f32 v64, v76, v186 :: v_dual_mul_f32 v65, v87, v187
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v74, v36, v194 :: v_dual_mul_f32 v75, v47, v195
	v_dual_mul_f32 v84, v56, v194 :: v_dual_mul_f32 v85, v67, v195
	v_dual_mul_f32 v94, v76, v194 :: v_dual_mul_f32 v95, v87, v195
	v_dual_fmac_f32 v44, v37, v187 :: v_dual_fmac_f32 v45, v46, v186
	v_dual_fmac_f32 v54, v57, v187 :: v_dual_fmac_f32 v55, v66, v186
	v_dual_fmac_f32 v64, v77, v187 :: v_dual_fmac_f32 v65, v86, v186
	v_dual_fmac_f32 v74, v37, v195 :: v_dual_fmac_f32 v75, v46, v194
	v_dual_fmac_f32 v84, v57, v195 :: v_dual_fmac_f32 v85, v66, v194
	v_dual_fmac_f32 v94, v77, v195 :: v_dual_fmac_f32 v95, v86, v194
	v_dual_fmac_f32 v44, v38, v188 :: v_dual_fmac_f32 v45, v48, v188
	v_dual_fmac_f32 v54, v58, v188 :: v_dual_fmac_f32 v55, v68, v188
	v_dual_fmac_f32 v64, v78, v188 :: v_dual_fmac_f32 v65, v88, v188
	v_dual_fmac_f32 v74, v38, v196 :: v_dual_fmac_f32 v75, v48, v196
	v_dual_fmac_f32 v84, v58, v196 :: v_dual_fmac_f32 v85, v68, v196
	v_dual_fmac_f32 v94, v78, v196 :: v_dual_fmac_f32 v95, v88, v196
	v_dual_fmac_f32 v44, v39, v189 :: v_dual_fmac_f32 v45, v49, v189
	v_dual_fmac_f32 v54, v59, v189 :: v_dual_fmac_f32 v55, v69, v189
	v_dual_fmac_f32 v64, v79, v189 :: v_dual_fmac_f32 v65, v89, v189
	v_dual_fmac_f32 v74, v39, v197 :: v_dual_fmac_f32 v75, v49, v197
	v_dual_fmac_f32 v84, v59, v197 :: v_dual_fmac_f32 v85, v69, v197
	v_dual_fmac_f32 v94, v79, v197 :: v_dual_fmac_f32 v95, v89, v197
	v_dual_fmac_f32 v44, v40, v190 :: v_dual_fmac_f32 v45, v50, v190
	v_dual_fmac_f32 v54, v60, v190 :: v_dual_fmac_f32 v55, v70, v190
	v_dual_fmac_f32 v64, v80, v190 :: v_dual_fmac_f32 v65, v90, v190
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v74, v40, v198 :: v_dual_fmac_f32 v75, v50, v198
	v_dual_fmac_f32 v84, v60, v198 :: v_dual_fmac_f32 v85, v70, v198
	v_dual_fmac_f32 v94, v80, v198 :: v_dual_fmac_f32 v95, v90, v198
	v_dual_fmac_f32 v44, v41, v191 :: v_dual_fmac_f32 v45, v51, v191
	v_dual_fmac_f32 v54, v61, v191 :: v_dual_fmac_f32 v55, v71, v191
	v_dual_fmac_f32 v64, v81, v191 :: v_dual_fmac_f32 v65, v91, v191
	v_dual_fmac_f32 v74, v41, v199 :: v_dual_fmac_f32 v75, v51, v199
	v_dual_fmac_f32 v84, v61, v199 :: v_dual_fmac_f32 v85, v71, v199
	v_dual_fmac_f32 v94, v81, v199 :: v_dual_fmac_f32 v95, v91, v199
	v_dual_fmac_f32 v44, v42, v192 :: v_dual_fmac_f32 v45, v52, v192
	v_dual_fmac_f32 v54, v62, v192 :: v_dual_fmac_f32 v55, v72, v192
	v_dual_fmac_f32 v64, v82, v192 :: v_dual_fmac_f32 v65, v92, v192
	v_dual_fmac_f32 v74, v42, v200 :: v_dual_fmac_f32 v75, v52, v200
	v_dual_fmac_f32 v84, v62, v200 :: v_dual_fmac_f32 v85, v72, v200
	v_dual_fmac_f32 v94, v82, v200 :: v_dual_fmac_f32 v95, v92, v200
	v_dual_fmac_f32 v44, v43, v193 :: v_dual_fmac_f32 v45, v53, v193
	v_dual_fmac_f32 v54, v63, v193 :: v_dual_fmac_f32 v55, v73, v193
	v_dual_fmac_f32 v64, v83, v193 :: v_dual_fmac_f32 v65, v93, v193
	v_dual_fmac_f32 v74, v43, v201 :: v_dual_fmac_f32 v75, v53, v201
	v_dual_fmac_f32 v84, v63, v201 :: v_dual_fmac_f32 v85, v73, v201
	v_dual_fmac_f32 v94, v83, v201 :: v_dual_fmac_f32 v95, v93, v201
	global_load_b128 v[186:189], v3, s[28:29]
	global_load_b128 v[190:193], v3, s[28:29] offset:16
	global_load_b128 v[194:197], v3, s[30:31]
	global_load_b128 v[198:201], v3, s[30:31] offset:16
	v_dual_add_f32 v108, v44, v108 :: v_dual_add_f32 v109, v109, v45
	v_dual_add_f32 v110, v54, v110 :: v_dual_add_f32 v111, v111, v55
	v_dual_add_f32 v112, v64, v112 :: v_dual_add_f32 v113, v113, v65
	v_dual_add_f32 v126, v74, v126 :: v_dual_add_f32 v127, v127, v75
	v_dual_add_f32 v128, v84, v128 :: v_dual_add_f32 v129, v129, v85
	v_dual_add_f32 v130, v94, v130 :: v_dual_add_f32 v131, v131, v95
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v44, v36, v202 :: v_dual_mul_f32 v45, v47, v203
	v_dual_mul_f32 v54, v56, v202 :: v_dual_mul_f32 v55, v67, v203
	v_dual_mul_f32 v64, v76, v202 :: v_dual_mul_f32 v65, v87, v203
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v74, v36, v210 :: v_dual_mul_f32 v75, v47, v211
	v_dual_mul_f32 v84, v56, v210 :: v_dual_mul_f32 v85, v67, v211
	v_dual_mul_f32 v94, v76, v210 :: v_dual_mul_f32 v95, v87, v211
	v_dual_fmac_f32 v44, v37, v203 :: v_dual_fmac_f32 v45, v46, v202
	v_dual_fmac_f32 v54, v57, v203 :: v_dual_fmac_f32 v55, v66, v202
	v_dual_fmac_f32 v64, v77, v203 :: v_dual_fmac_f32 v65, v86, v202
	v_dual_fmac_f32 v74, v37, v211 :: v_dual_fmac_f32 v75, v46, v210
	v_dual_fmac_f32 v84, v57, v211 :: v_dual_fmac_f32 v85, v66, v210
	v_dual_fmac_f32 v94, v77, v211 :: v_dual_fmac_f32 v95, v86, v210
	v_dual_fmac_f32 v44, v38, v204 :: v_dual_fmac_f32 v45, v48, v204
	v_dual_fmac_f32 v54, v58, v204 :: v_dual_fmac_f32 v55, v68, v204
	v_dual_fmac_f32 v64, v78, v204 :: v_dual_fmac_f32 v65, v88, v204
	v_dual_fmac_f32 v74, v38, v212 :: v_dual_fmac_f32 v75, v48, v212
	v_dual_fmac_f32 v84, v58, v212 :: v_dual_fmac_f32 v85, v68, v212
	v_dual_fmac_f32 v94, v78, v212 :: v_dual_fmac_f32 v95, v88, v212
	v_dual_fmac_f32 v44, v39, v205 :: v_dual_fmac_f32 v45, v49, v205
	v_dual_fmac_f32 v54, v59, v205 :: v_dual_fmac_f32 v55, v69, v205
	v_dual_fmac_f32 v64, v79, v205 :: v_dual_fmac_f32 v65, v89, v205
	v_dual_fmac_f32 v74, v39, v213 :: v_dual_fmac_f32 v75, v49, v213
	v_dual_fmac_f32 v84, v59, v213 :: v_dual_fmac_f32 v85, v69, v213
	v_dual_fmac_f32 v94, v79, v213 :: v_dual_fmac_f32 v95, v89, v213
	v_dual_fmac_f32 v44, v40, v206 :: v_dual_fmac_f32 v45, v50, v206
	v_dual_fmac_f32 v54, v60, v206 :: v_dual_fmac_f32 v55, v70, v206
	v_dual_fmac_f32 v64, v80, v206 :: v_dual_fmac_f32 v65, v90, v206
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v74, v40, v214 :: v_dual_fmac_f32 v75, v50, v214
	v_dual_fmac_f32 v84, v60, v214 :: v_dual_fmac_f32 v85, v70, v214
	v_dual_fmac_f32 v94, v80, v214 :: v_dual_fmac_f32 v95, v90, v214
	v_dual_fmac_f32 v44, v41, v207 :: v_dual_fmac_f32 v45, v51, v207
	v_dual_fmac_f32 v54, v61, v207 :: v_dual_fmac_f32 v55, v71, v207
	v_dual_fmac_f32 v64, v81, v207 :: v_dual_fmac_f32 v65, v91, v207
	v_dual_fmac_f32 v74, v41, v215 :: v_dual_fmac_f32 v75, v51, v215
	v_dual_fmac_f32 v84, v61, v215 :: v_dual_fmac_f32 v85, v71, v215
	v_dual_fmac_f32 v94, v81, v215 :: v_dual_fmac_f32 v95, v91, v215
	v_dual_fmac_f32 v44, v42, v208 :: v_dual_fmac_f32 v45, v52, v208
	v_dual_fmac_f32 v54, v62, v208 :: v_dual_fmac_f32 v55, v72, v208
	v_dual_fmac_f32 v64, v82, v208 :: v_dual_fmac_f32 v65, v92, v208
	v_dual_fmac_f32 v74, v42, v216 :: v_dual_fmac_f32 v75, v52, v216
	v_dual_fmac_f32 v84, v62, v216 :: v_dual_fmac_f32 v85, v72, v216
	v_dual_fmac_f32 v94, v82, v216 :: v_dual_fmac_f32 v95, v92, v216
	v_dual_fmac_f32 v44, v43, v209 :: v_dual_fmac_f32 v45, v53, v209
	v_dual_fmac_f32 v54, v63, v209 :: v_dual_fmac_f32 v55, v73, v209
	v_dual_fmac_f32 v64, v83, v209 :: v_dual_fmac_f32 v65, v93, v209
	v_dual_fmac_f32 v74, v43, v217 :: v_dual_fmac_f32 v75, v53, v217
	v_dual_fmac_f32 v84, v63, v217 :: v_dual_fmac_f32 v85, v73, v217
	v_dual_fmac_f32 v94, v83, v217 :: v_dual_fmac_f32 v95, v93, v217
	global_load_b128 v[202:205], v3, s[32:33]
	global_load_b128 v[206:209], v3, s[32:33] offset:16
	global_load_b128 v[210:213], v3, s[34:35]
	global_load_b128 v[214:217], v3, s[34:35] offset:16
	v_dual_add_f32 v144, v44, v144 :: v_dual_add_f32 v145, v145, v45
	v_dual_add_f32 v146, v54, v146 :: v_dual_add_f32 v147, v147, v55
	v_dual_add_f32 v148, v64, v148 :: v_dual_add_f32 v149, v149, v65
	v_dual_add_f32 v162, v74, v162 :: v_dual_add_f32 v163, v163, v75
	v_dual_add_f32 v164, v84, v164 :: v_dual_add_f32 v165, v165, v85
	v_dual_add_f32 v166, v94, v166 :: v_dual_add_f32 v167, v167, v95
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v44, v36, v218 :: v_dual_mul_f32 v45, v47, v219
	v_dual_mul_f32 v54, v56, v218 :: v_dual_mul_f32 v55, v67, v219
	v_dual_mul_f32 v64, v76, v218 :: v_dual_mul_f32 v65, v87, v219
	v_dual_fmac_f32 v44, v37, v219 :: v_dual_fmac_f32 v45, v46, v218
	v_dual_fmac_f32 v54, v57, v219 :: v_dual_fmac_f32 v55, v66, v218
	v_dual_fmac_f32 v64, v77, v219 :: v_dual_fmac_f32 v65, v86, v218
	v_dual_fmac_f32 v44, v38, v220 :: v_dual_fmac_f32 v45, v48, v220
	v_dual_fmac_f32 v54, v58, v220 :: v_dual_fmac_f32 v55, v68, v220
	v_dual_fmac_f32 v64, v78, v220 :: v_dual_fmac_f32 v65, v88, v220
	v_dual_fmac_f32 v44, v39, v221 :: v_dual_fmac_f32 v45, v49, v221
	v_dual_fmac_f32 v54, v59, v221 :: v_dual_fmac_f32 v55, v69, v221
	v_dual_fmac_f32 v64, v79, v221 :: v_dual_fmac_f32 v65, v89, v221
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v44, v40, v222 :: v_dual_fmac_f32 v45, v50, v222
	v_dual_fmac_f32 v54, v60, v222 :: v_dual_fmac_f32 v55, v70, v222
	v_dual_fmac_f32 v64, v80, v222 :: v_dual_fmac_f32 v65, v90, v222
	v_dual_fmac_f32 v44, v41, v223 :: v_dual_fmac_f32 v45, v51, v223
	v_dual_fmac_f32 v54, v61, v223 :: v_dual_fmac_f32 v55, v71, v223
	v_dual_fmac_f32 v64, v81, v223 :: v_dual_fmac_f32 v65, v91, v223
	v_dual_fmac_f32 v44, v42, v224 :: v_dual_fmac_f32 v45, v52, v224
	v_dual_fmac_f32 v54, v62, v224 :: v_dual_fmac_f32 v55, v72, v224
	v_dual_fmac_f32 v64, v82, v224 :: v_dual_fmac_f32 v65, v92, v224
	v_dual_fmac_f32 v44, v43, v225 :: v_dual_fmac_f32 v45, v53, v225
	v_dual_fmac_f32 v54, v63, v225 :: v_dual_fmac_f32 v55, v73, v225
	v_dual_fmac_f32 v64, v83, v225 :: v_dual_fmac_f32 v65, v93, v225
	global_load_b128 v[218:221], v3, s[36:37]
	global_load_b128 v[222:225], v3, s[36:37] offset:16
	v_dual_add_f32 v180, v44, v180 :: v_dual_add_f32 v181, v181, v45
	v_dual_add_f32 v182, v54, v182 :: v_dual_add_f32 v183, v183, v55
	v_dual_add_f32 v184, v64, v184 :: v_dual_add_f32 v185, v185, v65
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s3
	s_wait_loadcnt 0x0
	.Lrx_b5_s3_done:
	v_dual_add_f32 v102, v102, v108 :: v_dual_add_f32 v103, v103, v109
	v_dual_add_f32 v104, v104, v110 :: v_dual_add_f32 v105, v105, v111
	v_dual_add_f32 v106, v106, v112 :: v_dual_add_f32 v107, v107, v113
	v_dual_add_f32 v120, v120, v126 :: v_dual_add_f32 v121, v121, v127
	v_dual_add_f32 v122, v122, v128 :: v_dual_add_f32 v123, v123, v129
	v_dual_add_f32 v124, v124, v130 :: v_dual_add_f32 v125, v125, v131
	v_dual_add_f32 v138, v138, v144 :: v_dual_add_f32 v139, v139, v145
	v_dual_add_f32 v140, v140, v146 :: v_dual_add_f32 v141, v141, v147
	v_dual_add_f32 v142, v142, v148 :: v_dual_add_f32 v143, v143, v149
	v_dual_add_f32 v156, v156, v162 :: v_dual_add_f32 v157, v157, v163
	v_dual_add_f32 v158, v158, v164 :: v_dual_add_f32 v159, v159, v165
	v_dual_add_f32 v160, v160, v166 :: v_dual_add_f32 v161, v161, v167
	v_dual_add_f32 v174, v174, v180 :: v_dual_add_f32 v175, v175, v181
	v_dual_add_f32 v176, v176, v182 :: v_dual_add_f32 v177, v177, v183
	v_dual_add_f32 v178, v178, v184 :: v_dual_add_f32 v179, v179, v185
	v_dual_add_f32 v96, v96, v102 :: v_dual_add_f32 v97, v97, v103
	v_dual_add_f32 v98, v98, v104 :: v_dual_add_f32 v99, v99, v105
	v_dual_add_f32 v100, v100, v106 :: v_dual_add_f32 v101, v101, v107
	v_dual_add_f32 v114, v114, v120 :: v_dual_add_f32 v115, v115, v121
	v_dual_add_f32 v116, v116, v122 :: v_dual_add_f32 v117, v117, v123
	v_dual_add_f32 v118, v118, v124 :: v_dual_add_f32 v119, v119, v125
	v_dual_add_f32 v132, v132, v138 :: v_dual_add_f32 v133, v133, v139
	v_dual_add_f32 v134, v134, v140 :: v_dual_add_f32 v135, v135, v141
	v_dual_add_f32 v136, v136, v142 :: v_dual_add_f32 v137, v137, v143
	v_dual_add_f32 v150, v150, v156 :: v_dual_add_f32 v151, v151, v157
	v_dual_add_f32 v152, v152, v158 :: v_dual_add_f32 v153, v153, v159
	v_dual_add_f32 v154, v154, v160 :: v_dual_add_f32 v155, v155, v161
	v_dual_add_f32 v168, v168, v174 :: v_dual_add_f32 v169, v169, v175
	v_dual_add_f32 v170, v170, v176 :: v_dual_add_f32 v171, v171, v177
	v_dual_add_f32 v172, v172, v178 :: v_dual_add_f32 v173, v173, v179
	ds_swizzle_b32 v186, v96 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v187, v97 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v188, v98 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v189, v99 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v190, v100 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v191, v101 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v192, v114 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v193, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v194, v116 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v195, v117 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v196, v118 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v197, v119 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v198, v132 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v199, v133 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v200, v134 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v201, v135 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v202, v136 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v203, v137 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v204, v150 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v205, v151 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v206, v152 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v207, v153 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v208, v154 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v209, v155 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v210, v168 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v211, v169 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v212, v170 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v213, v171 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v214, v172 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v215, v173 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x1d
	v_add_f32_e32 v96, v96, v186
	s_wait_dscnt 0x1c
	v_add_f32_e32 v97, v97, v187
	s_wait_dscnt 0x1b
	v_add_f32_e32 v98, v98, v188
	s_wait_dscnt 0x1a
	v_add_f32_e32 v99, v99, v189
	s_wait_dscnt 0x19
	v_add_f32_e32 v100, v100, v190
	s_wait_dscnt 0x18
	v_add_f32_e32 v101, v101, v191
	s_wait_dscnt 0x17
	v_add_f32_e32 v114, v114, v192
	s_wait_dscnt 0x16
	v_add_f32_e32 v115, v115, v193
	s_wait_dscnt 0x15
	v_add_f32_e32 v116, v116, v194
	s_wait_dscnt 0x14
	v_add_f32_e32 v117, v117, v195
	s_wait_dscnt 0x13
	v_add_f32_e32 v118, v118, v196
	s_wait_dscnt 0x12
	v_add_f32_e32 v119, v119, v197
	s_wait_dscnt 0x11
	v_add_f32_e32 v132, v132, v198
	s_wait_dscnt 0x10
	v_add_f32_e32 v133, v133, v199
	s_wait_dscnt 0xf
	v_add_f32_e32 v134, v134, v200
	s_wait_dscnt 0xe
	v_add_f32_e32 v135, v135, v201
	s_wait_dscnt 0xd
	v_add_f32_e32 v136, v136, v202
	s_wait_dscnt 0xc
	v_add_f32_e32 v137, v137, v203
	s_wait_dscnt 0xb
	v_add_f32_e32 v150, v150, v204
	s_wait_dscnt 0xa
	v_add_f32_e32 v151, v151, v205
	s_wait_dscnt 0x9
	v_add_f32_e32 v152, v152, v206
	s_wait_dscnt 0x8
	v_add_f32_e32 v153, v153, v207
	s_wait_dscnt 0x7
	v_add_f32_e32 v154, v154, v208
	s_wait_dscnt 0x6
	v_add_f32_e32 v155, v155, v209
	s_wait_dscnt 0x5
	v_add_f32_e32 v168, v168, v210
	s_wait_dscnt 0x4
	v_add_f32_e32 v169, v169, v211
	s_wait_dscnt 0x3
	v_add_f32_e32 v170, v170, v212
	s_wait_dscnt 0x2
	v_add_f32_e32 v171, v171, v213
	s_wait_dscnt 0x1
	v_add_f32_e32 v172, v172, v214
	s_wait_dscnt 0x0
	v_add_f32_e32 v173, v173, v215
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v186, v3, v96
	ds_bpermute_b32 v187, v3, v97
	ds_bpermute_b32 v188, v3, v98
	ds_bpermute_b32 v189, v3, v99
	ds_bpermute_b32 v190, v3, v100
	ds_bpermute_b32 v191, v3, v101
	ds_bpermute_b32 v192, v3, v114
	ds_bpermute_b32 v193, v3, v115
	ds_bpermute_b32 v194, v3, v116
	ds_bpermute_b32 v195, v3, v117
	ds_bpermute_b32 v196, v3, v118
	ds_bpermute_b32 v197, v3, v119
	ds_bpermute_b32 v198, v3, v132
	ds_bpermute_b32 v199, v3, v133
	ds_bpermute_b32 v200, v3, v134
	ds_bpermute_b32 v201, v3, v135
	ds_bpermute_b32 v202, v3, v136
	ds_bpermute_b32 v203, v3, v137
	ds_bpermute_b32 v204, v3, v150
	ds_bpermute_b32 v205, v3, v151
	ds_bpermute_b32 v206, v3, v152
	ds_bpermute_b32 v207, v3, v153
	ds_bpermute_b32 v208, v3, v154
	ds_bpermute_b32 v209, v3, v155
	ds_bpermute_b32 v210, v3, v168
	ds_bpermute_b32 v211, v3, v169
	ds_bpermute_b32 v212, v3, v170
	ds_bpermute_b32 v213, v3, v171
	ds_bpermute_b32 v214, v3, v172
	ds_bpermute_b32 v215, v3, v173
	s_wait_dscnt 0x1d
	v_add_f32_e32 v96, v96, v186
	s_wait_dscnt 0x1c
	v_add_f32_e32 v97, v97, v187
	s_wait_dscnt 0x1b
	v_add_f32_e32 v98, v98, v188
	s_wait_dscnt 0x1a
	v_add_f32_e32 v99, v99, v189
	s_wait_dscnt 0x19
	v_add_f32_e32 v100, v100, v190
	s_wait_dscnt 0x18
	v_add_f32_e32 v101, v101, v191
	s_wait_dscnt 0x17
	v_add_f32_e32 v114, v114, v192
	s_wait_dscnt 0x16
	v_add_f32_e32 v115, v115, v193
	s_wait_dscnt 0x15
	v_add_f32_e32 v116, v116, v194
	s_wait_dscnt 0x14
	v_add_f32_e32 v117, v117, v195
	s_wait_dscnt 0x13
	v_add_f32_e32 v118, v118, v196
	s_wait_dscnt 0x12
	v_add_f32_e32 v119, v119, v197
	s_wait_dscnt 0x11
	v_add_f32_e32 v132, v132, v198
	s_wait_dscnt 0x10
	v_add_f32_e32 v133, v133, v199
	s_wait_dscnt 0xf
	v_add_f32_e32 v134, v134, v200
	s_wait_dscnt 0xe
	v_add_f32_e32 v135, v135, v201
	s_wait_dscnt 0xd
	v_add_f32_e32 v136, v136, v202
	s_wait_dscnt 0xc
	v_add_f32_e32 v137, v137, v203
	s_wait_dscnt 0xb
	v_add_f32_e32 v150, v150, v204
	s_wait_dscnt 0xa
	v_add_f32_e32 v151, v151, v205
	s_wait_dscnt 0x9
	v_add_f32_e32 v152, v152, v206
	s_wait_dscnt 0x8
	v_add_f32_e32 v153, v153, v207
	s_wait_dscnt 0x7
	v_add_f32_e32 v154, v154, v208
	s_wait_dscnt 0x6
	v_add_f32_e32 v155, v155, v209
	s_wait_dscnt 0x5
	v_add_f32_e32 v168, v168, v210
	s_wait_dscnt 0x4
	v_add_f32_e32 v169, v169, v211
	s_wait_dscnt 0x3
	v_add_f32_e32 v170, v170, v212
	s_wait_dscnt 0x2
	v_add_f32_e32 v171, v171, v213
	s_wait_dscnt 0x1
	v_add_f32_e32 v172, v172, v214
	s_wait_dscnt 0x0
	v_add_f32_e32 v173, v173, v215
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v186, v3, v96
	ds_bpermute_b32 v187, v3, v97
	ds_bpermute_b32 v188, v3, v98
	ds_bpermute_b32 v189, v3, v99
	ds_bpermute_b32 v190, v3, v100
	ds_bpermute_b32 v191, v3, v101
	ds_bpermute_b32 v192, v3, v114
	ds_bpermute_b32 v193, v3, v115
	ds_bpermute_b32 v194, v3, v116
	ds_bpermute_b32 v195, v3, v117
	ds_bpermute_b32 v196, v3, v118
	ds_bpermute_b32 v197, v3, v119
	ds_bpermute_b32 v198, v3, v132
	ds_bpermute_b32 v199, v3, v133
	ds_bpermute_b32 v200, v3, v134
	ds_bpermute_b32 v201, v3, v135
	ds_bpermute_b32 v202, v3, v136
	ds_bpermute_b32 v203, v3, v137
	ds_bpermute_b32 v204, v3, v150
	ds_bpermute_b32 v205, v3, v151
	ds_bpermute_b32 v206, v3, v152
	ds_bpermute_b32 v207, v3, v153
	ds_bpermute_b32 v208, v3, v154
	ds_bpermute_b32 v209, v3, v155
	ds_bpermute_b32 v210, v3, v168
	ds_bpermute_b32 v211, v3, v169
	ds_bpermute_b32 v212, v3, v170
	ds_bpermute_b32 v213, v3, v171
	ds_bpermute_b32 v214, v3, v172
	ds_bpermute_b32 v215, v3, v173
	s_wait_dscnt 0x1d
	v_add_f32_e32 v96, v96, v186
	s_wait_dscnt 0x1c
	v_add_f32_e32 v97, v97, v187
	s_wait_dscnt 0x1b
	v_add_f32_e32 v98, v98, v188
	s_wait_dscnt 0x1a
	v_add_f32_e32 v99, v99, v189
	s_wait_dscnt 0x19
	v_add_f32_e32 v100, v100, v190
	s_wait_dscnt 0x18
	v_add_f32_e32 v101, v101, v191
	s_wait_dscnt 0x17
	v_add_f32_e32 v114, v114, v192
	s_wait_dscnt 0x16
	v_add_f32_e32 v115, v115, v193
	s_wait_dscnt 0x15
	v_add_f32_e32 v116, v116, v194
	s_wait_dscnt 0x14
	v_add_f32_e32 v117, v117, v195
	s_wait_dscnt 0x13
	v_add_f32_e32 v118, v118, v196
	s_wait_dscnt 0x12
	v_add_f32_e32 v119, v119, v197
	s_wait_dscnt 0x11
	v_add_f32_e32 v132, v132, v198
	s_wait_dscnt 0x10
	v_add_f32_e32 v133, v133, v199
	s_wait_dscnt 0xf
	v_add_f32_e32 v134, v134, v200
	s_wait_dscnt 0xe
	v_add_f32_e32 v135, v135, v201
	s_wait_dscnt 0xd
	v_add_f32_e32 v136, v136, v202
	s_wait_dscnt 0xc
	v_add_f32_e32 v137, v137, v203
	s_wait_dscnt 0xb
	v_add_f32_e32 v150, v150, v204
	s_wait_dscnt 0xa
	v_add_f32_e32 v151, v151, v205
	s_wait_dscnt 0x9
	v_add_f32_e32 v152, v152, v206
	s_wait_dscnt 0x8
	v_add_f32_e32 v153, v153, v207
	s_wait_dscnt 0x7
	v_add_f32_e32 v154, v154, v208
	s_wait_dscnt 0x6
	v_add_f32_e32 v155, v155, v209
	s_wait_dscnt 0x5
	v_add_f32_e32 v168, v168, v210
	s_wait_dscnt 0x4
	v_add_f32_e32 v169, v169, v211
	s_wait_dscnt 0x3
	v_add_f32_e32 v170, v170, v212
	s_wait_dscnt 0x2
	v_add_f32_e32 v171, v171, v213
	s_wait_dscnt 0x1
	v_add_f32_e32 v172, v172, v214
	s_wait_dscnt 0x0
	v_add_f32_e32 v173, v173, v215
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v186, v3, v96
	ds_bpermute_b32 v187, v3, v97
	ds_bpermute_b32 v188, v3, v98
	ds_bpermute_b32 v189, v3, v99
	ds_bpermute_b32 v190, v3, v100
	ds_bpermute_b32 v191, v3, v101
	ds_bpermute_b32 v192, v3, v114
	ds_bpermute_b32 v193, v3, v115
	ds_bpermute_b32 v194, v3, v116
	ds_bpermute_b32 v195, v3, v117
	ds_bpermute_b32 v196, v3, v118
	ds_bpermute_b32 v197, v3, v119
	ds_bpermute_b32 v198, v3, v132
	ds_bpermute_b32 v199, v3, v133
	ds_bpermute_b32 v200, v3, v134
	ds_bpermute_b32 v201, v3, v135
	ds_bpermute_b32 v202, v3, v136
	ds_bpermute_b32 v203, v3, v137
	ds_bpermute_b32 v204, v3, v150
	ds_bpermute_b32 v205, v3, v151
	ds_bpermute_b32 v206, v3, v152
	ds_bpermute_b32 v207, v3, v153
	ds_bpermute_b32 v208, v3, v154
	ds_bpermute_b32 v209, v3, v155
	ds_bpermute_b32 v210, v3, v168
	ds_bpermute_b32 v211, v3, v169
	ds_bpermute_b32 v212, v3, v170
	ds_bpermute_b32 v213, v3, v171
	ds_bpermute_b32 v214, v3, v172
	ds_bpermute_b32 v215, v3, v173
	s_wait_dscnt 0x1d
	v_add_f32_e32 v96, v96, v186
	s_wait_dscnt 0x1c
	v_add_f32_e32 v97, v97, v187
	s_wait_dscnt 0x1b
	v_add_f32_e32 v98, v98, v188
	s_wait_dscnt 0x1a
	v_add_f32_e32 v99, v99, v189
	s_wait_dscnt 0x19
	v_add_f32_e32 v100, v100, v190
	s_wait_dscnt 0x18
	v_add_f32_e32 v101, v101, v191
	s_wait_dscnt 0x17
	v_add_f32_e32 v114, v114, v192
	s_wait_dscnt 0x16
	v_add_f32_e32 v115, v115, v193
	s_wait_dscnt 0x15
	v_add_f32_e32 v116, v116, v194
	s_wait_dscnt 0x14
	v_add_f32_e32 v117, v117, v195
	s_wait_dscnt 0x13
	v_add_f32_e32 v118, v118, v196
	s_wait_dscnt 0x12
	v_add_f32_e32 v119, v119, v197
	s_wait_dscnt 0x11
	v_add_f32_e32 v132, v132, v198
	s_wait_dscnt 0x10
	v_add_f32_e32 v133, v133, v199
	s_wait_dscnt 0xf
	v_add_f32_e32 v134, v134, v200
	s_wait_dscnt 0xe
	v_add_f32_e32 v135, v135, v201
	s_wait_dscnt 0xd
	v_add_f32_e32 v136, v136, v202
	s_wait_dscnt 0xc
	v_add_f32_e32 v137, v137, v203
	s_wait_dscnt 0xb
	v_add_f32_e32 v150, v150, v204
	s_wait_dscnt 0xa
	v_add_f32_e32 v151, v151, v205
	s_wait_dscnt 0x9
	v_add_f32_e32 v152, v152, v206
	s_wait_dscnt 0x8
	v_add_f32_e32 v153, v153, v207
	s_wait_dscnt 0x7
	v_add_f32_e32 v154, v154, v208
	s_wait_dscnt 0x6
	v_add_f32_e32 v155, v155, v209
	s_wait_dscnt 0x5
	v_add_f32_e32 v168, v168, v210
	s_wait_dscnt 0x4
	v_add_f32_e32 v169, v169, v211
	s_wait_dscnt 0x3
	v_add_f32_e32 v170, v170, v212
	s_wait_dscnt 0x2
	v_add_f32_e32 v171, v171, v213
	s_wait_dscnt 0x1
	v_add_f32_e32 v172, v172, v214
	s_wait_dscnt 0x0
	v_add_f32_e32 v173, v173, v215
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v186, v3, v96
	ds_bpermute_b32 v187, v3, v97
	ds_bpermute_b32 v188, v3, v98
	ds_bpermute_b32 v189, v3, v99
	ds_bpermute_b32 v190, v3, v100
	ds_bpermute_b32 v191, v3, v101
	ds_bpermute_b32 v192, v3, v114
	ds_bpermute_b32 v193, v3, v115
	ds_bpermute_b32 v194, v3, v116
	ds_bpermute_b32 v195, v3, v117
	ds_bpermute_b32 v196, v3, v118
	ds_bpermute_b32 v197, v3, v119
	ds_bpermute_b32 v198, v3, v132
	ds_bpermute_b32 v199, v3, v133
	ds_bpermute_b32 v200, v3, v134
	ds_bpermute_b32 v201, v3, v135
	ds_bpermute_b32 v202, v3, v136
	ds_bpermute_b32 v203, v3, v137
	ds_bpermute_b32 v204, v3, v150
	ds_bpermute_b32 v205, v3, v151
	ds_bpermute_b32 v206, v3, v152
	ds_bpermute_b32 v207, v3, v153
	ds_bpermute_b32 v208, v3, v154
	ds_bpermute_b32 v209, v3, v155
	ds_bpermute_b32 v210, v3, v168
	ds_bpermute_b32 v211, v3, v169
	ds_bpermute_b32 v212, v3, v170
	ds_bpermute_b32 v213, v3, v171
	ds_bpermute_b32 v214, v3, v172
	ds_bpermute_b32 v215, v3, v173
	s_wait_dscnt 0x1d
	v_add_f32_e32 v96, v96, v186
	s_wait_dscnt 0x1c
	v_add_f32_e32 v97, v97, v187
	s_wait_dscnt 0x1b
	v_add_f32_e32 v98, v98, v188
	s_wait_dscnt 0x1a
	v_add_f32_e32 v99, v99, v189
	s_wait_dscnt 0x19
	v_add_f32_e32 v100, v100, v190
	s_wait_dscnt 0x18
	v_add_f32_e32 v101, v101, v191
	s_wait_dscnt 0x17
	v_add_f32_e32 v114, v114, v192
	s_wait_dscnt 0x16
	v_add_f32_e32 v115, v115, v193
	s_wait_dscnt 0x15
	v_add_f32_e32 v116, v116, v194
	s_wait_dscnt 0x14
	v_add_f32_e32 v117, v117, v195
	s_wait_dscnt 0x13
	v_add_f32_e32 v118, v118, v196
	s_wait_dscnt 0x12
	v_add_f32_e32 v119, v119, v197
	s_wait_dscnt 0x11
	v_add_f32_e32 v132, v132, v198
	s_wait_dscnt 0x10
	v_add_f32_e32 v133, v133, v199
	s_wait_dscnt 0xf
	v_add_f32_e32 v134, v134, v200
	s_wait_dscnt 0xe
	v_add_f32_e32 v135, v135, v201
	s_wait_dscnt 0xd
	v_add_f32_e32 v136, v136, v202
	s_wait_dscnt 0xc
	v_add_f32_e32 v137, v137, v203
	s_wait_dscnt 0xb
	v_add_f32_e32 v150, v150, v204
	s_wait_dscnt 0xa
	v_add_f32_e32 v151, v151, v205
	s_wait_dscnt 0x9
	v_add_f32_e32 v152, v152, v206
	s_wait_dscnt 0x8
	v_add_f32_e32 v153, v153, v207
	s_wait_dscnt 0x7
	v_add_f32_e32 v154, v154, v208
	s_wait_dscnt 0x6
	v_add_f32_e32 v155, v155, v209
	s_wait_dscnt 0x5
	v_add_f32_e32 v168, v168, v210
	s_wait_dscnt 0x4
	v_add_f32_e32 v169, v169, v211
	s_wait_dscnt 0x3
	v_add_f32_e32 v170, v170, v212
	s_wait_dscnt 0x2
	v_add_f32_e32 v171, v171, v213
	s_wait_dscnt 0x1
	v_add_f32_e32 v172, v172, v214
	s_wait_dscnt 0x0
	v_add_f32_e32 v173, v173, v215
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v66, s46
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v67, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v68, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v69, s47
	s_mul_i32 s47, s45, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v70, s47
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b5_p0_single
	global_load_b32 v36, v66, s[8:9] offset:0
	global_load_b32 v37, v66, s[8:9] offset:4
	global_load_b32 v42, v67, s[8:9] offset:0
	global_load_b32 v43, v67, s[8:9] offset:4
	global_load_b32 v48, v68, s[8:9] offset:0
	global_load_b32 v49, v68, s[8:9] offset:4
	global_load_b32 v54, v69, s[8:9] offset:0
	global_load_b32 v55, v69, s[8:9] offset:4
	global_load_b32 v60, v70, s[8:9] offset:0
	global_load_b32 v61, v70, s[8:9] offset:4
	s_wait_loadcnt 0x9
	v_add_f32_e32 v36, v96, v36
	global_store_b32 v66, v36, s[8:9] offset:0
	s_wait_loadcnt 0x8
	v_add_f32_e32 v37, v97, v37
	global_store_b32 v66, v37, s[8:9] offset:4
	s_wait_loadcnt 0x7
	v_add_f32_e32 v42, v114, v42
	global_store_b32 v67, v42, s[8:9] offset:0
	s_wait_loadcnt 0x6
	v_add_f32_e32 v43, v115, v43
	global_store_b32 v67, v43, s[8:9] offset:4
	s_wait_loadcnt 0x5
	v_add_f32_e32 v48, v132, v48
	global_store_b32 v68, v48, s[8:9] offset:0
	s_wait_loadcnt 0x4
	v_add_f32_e32 v49, v133, v49
	global_store_b32 v68, v49, s[8:9] offset:4
	s_wait_loadcnt 0x3
	v_add_f32_e32 v54, v150, v54
	global_store_b32 v69, v54, s[8:9] offset:0
	s_wait_loadcnt 0x2
	v_add_f32_e32 v55, v151, v55
	global_store_b32 v69, v55, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v60, v168, v60
	global_store_b32 v70, v60, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v61, v169, v61
	global_store_b32 v70, v61, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b5_p0_next
	.Lrx_b5_p0_single:
	global_load_b32 v36, v66, s[8:9] offset:0
	global_load_b32 v42, v67, s[8:9] offset:0
	global_load_b32 v48, v68, s[8:9] offset:0
	global_load_b32 v54, v69, s[8:9] offset:0
	global_load_b32 v60, v70, s[8:9] offset:0
	s_wait_loadcnt 0x4
	v_add_f32_e32 v36, v36, v96
	global_store_b32 v66, v36, s[8:9] offset:0
	s_wait_loadcnt 0x3
	v_add_f32_e32 v42, v42, v114
	global_store_b32 v67, v42, s[8:9] offset:0
	s_wait_loadcnt 0x2
	v_add_f32_e32 v48, v48, v132
	global_store_b32 v68, v48, s[8:9] offset:0
	s_wait_loadcnt 0x1
	v_add_f32_e32 v54, v54, v150
	global_store_b32 v69, v54, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v60, v60, v168
	global_store_b32 v70, v60, s[8:9] offset:0
	s_wait_storecnt 0x0
	s_branch .Lrx_b5_stored
	.Lrx_b5_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b5_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b5_p1_single
	global_load_b32 v38, v66, s[8:9] offset:8
	global_load_b32 v39, v66, s[8:9] offset:12
	global_load_b32 v44, v67, s[8:9] offset:8
	global_load_b32 v45, v67, s[8:9] offset:12
	global_load_b32 v50, v68, s[8:9] offset:8
	global_load_b32 v51, v68, s[8:9] offset:12
	global_load_b32 v56, v69, s[8:9] offset:8
	global_load_b32 v57, v69, s[8:9] offset:12
	global_load_b32 v62, v70, s[8:9] offset:8
	global_load_b32 v63, v70, s[8:9] offset:12
	s_wait_loadcnt 0x9
	v_add_f32_e32 v38, v98, v38
	global_store_b32 v66, v38, s[8:9] offset:8
	s_wait_loadcnt 0x8
	v_add_f32_e32 v39, v99, v39
	global_store_b32 v66, v39, s[8:9] offset:12
	s_wait_loadcnt 0x7
	v_add_f32_e32 v44, v116, v44
	global_store_b32 v67, v44, s[8:9] offset:8
	s_wait_loadcnt 0x6
	v_add_f32_e32 v45, v117, v45
	global_store_b32 v67, v45, s[8:9] offset:12
	s_wait_loadcnt 0x5
	v_add_f32_e32 v50, v134, v50
	global_store_b32 v68, v50, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v51, v135, v51
	global_store_b32 v68, v51, s[8:9] offset:12
	s_wait_loadcnt 0x3
	v_add_f32_e32 v56, v152, v56
	global_store_b32 v69, v56, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v57, v153, v57
	global_store_b32 v69, v57, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v62, v170, v62
	global_store_b32 v70, v62, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v63, v171, v63
	global_store_b32 v70, v63, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b5_p1_next
	.Lrx_b5_p1_single:
	global_load_b32 v38, v66, s[8:9] offset:8
	global_load_b32 v44, v67, s[8:9] offset:8
	global_load_b32 v50, v68, s[8:9] offset:8
	global_load_b32 v56, v69, s[8:9] offset:8
	global_load_b32 v62, v70, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v38, v38, v98
	global_store_b32 v66, v38, s[8:9] offset:8
	s_wait_loadcnt 0x3
	v_add_f32_e32 v44, v44, v116
	global_store_b32 v67, v44, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v50, v50, v134
	global_store_b32 v68, v50, s[8:9] offset:8
	s_wait_loadcnt 0x1
	v_add_f32_e32 v56, v56, v152
	global_store_b32 v69, v56, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v62, v62, v170
	global_store_b32 v70, v62, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b5_stored
	.Lrx_b5_p1_next:
	s_add_co_i32 s13, s12, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b5_stored
	s_add_co_i32 s13, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b5_p2_single
	global_load_b32 v40, v66, s[8:9] offset:16
	global_load_b32 v41, v66, s[8:9] offset:20
	global_load_b32 v46, v67, s[8:9] offset:16
	global_load_b32 v47, v67, s[8:9] offset:20
	global_load_b32 v52, v68, s[8:9] offset:16
	global_load_b32 v53, v68, s[8:9] offset:20
	global_load_b32 v58, v69, s[8:9] offset:16
	global_load_b32 v59, v69, s[8:9] offset:20
	global_load_b32 v64, v70, s[8:9] offset:16
	global_load_b32 v65, v70, s[8:9] offset:20
	s_wait_loadcnt 0x9
	v_add_f32_e32 v40, v100, v40
	global_store_b32 v66, v40, s[8:9] offset:16
	s_wait_loadcnt 0x8
	v_add_f32_e32 v41, v101, v41
	global_store_b32 v66, v41, s[8:9] offset:20
	s_wait_loadcnt 0x7
	v_add_f32_e32 v46, v118, v46
	global_store_b32 v67, v46, s[8:9] offset:16
	s_wait_loadcnt 0x6
	v_add_f32_e32 v47, v119, v47
	global_store_b32 v67, v47, s[8:9] offset:20
	s_wait_loadcnt 0x5
	v_add_f32_e32 v52, v136, v52
	global_store_b32 v68, v52, s[8:9] offset:16
	s_wait_loadcnt 0x4
	v_add_f32_e32 v53, v137, v53
	global_store_b32 v68, v53, s[8:9] offset:20
	s_wait_loadcnt 0x3
	v_add_f32_e32 v58, v154, v58
	global_store_b32 v69, v58, s[8:9] offset:16
	s_wait_loadcnt 0x2
	v_add_f32_e32 v59, v155, v59
	global_store_b32 v69, v59, s[8:9] offset:20
	s_wait_loadcnt 0x1
	v_add_f32_e32 v64, v172, v64
	global_store_b32 v70, v64, s[8:9] offset:16
	s_wait_loadcnt 0x0
	v_add_f32_e32 v65, v173, v65
	global_store_b32 v70, v65, s[8:9] offset:20
	s_wait_storecnt 0x0
	s_branch .Lrx_b5_stored
	.Lrx_b5_p2_single:
	global_load_b32 v40, v66, s[8:9] offset:16
	global_load_b32 v46, v67, s[8:9] offset:16
	global_load_b32 v52, v68, s[8:9] offset:16
	global_load_b32 v58, v69, s[8:9] offset:16
	global_load_b32 v64, v70, s[8:9] offset:16
	s_wait_loadcnt 0x4
	v_add_f32_e32 v40, v40, v100
	global_store_b32 v66, v40, s[8:9] offset:16
	s_wait_loadcnt 0x3
	v_add_f32_e32 v46, v46, v118
	global_store_b32 v67, v46, s[8:9] offset:16
	s_wait_loadcnt 0x2
	v_add_f32_e32 v52, v52, v136
	global_store_b32 v68, v52, s[8:9] offset:16
	s_wait_loadcnt 0x1
	v_add_f32_e32 v58, v58, v154
	global_store_b32 v69, v58, s[8:9] offset:16
	s_wait_loadcnt 0x0
	v_add_f32_e32 v64, v64, v172
	global_store_b32 v70, v64, s[8:9] offset:16
	s_wait_storecnt 0x0
	s_branch .Lrx_b5_stored
	.Lrx_b5_stored:
	s_branch .Lrx_end
	.Lrx_b6:
	s_mul_i32 s12, ttmp9, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v12, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v13, s13
	v_add_nc_u32_e32 v7, s13, v1
	s_add_co_i32 s13, s12, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v14, s13
	v_add_nc_u32_e32 v8, s13, v1
	s_add_co_i32 s13, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v15, s13
	v_add_nc_u32_e32 v9, s13, v1
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s0_done
	s_mov_b32 s15, 0x0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[204:207], v2, s[28:29]
	global_load_b128 v[208:211], v2, s[28:29] offset:16
	global_load_b128 v[212:215], v2, s[30:31]
	global_load_b128 v[216:219], v2, s[30:31] offset:16
	global_load_b128 v[220:223], v2, s[32:33]
	global_load_b128 v[224:227], v2, s[32:33] offset:16
	global_load_b128 v[228:231], v2, s[34:35]
	global_load_b128 v[232:235], v2, s[34:35] offset:16
	global_load_b128 v[236:239], v2, s[36:37]
	global_load_b128 v[240:243], v2, s[36:37] offset:16
	global_load_b128 v[244:247], v2, s[38:39]
	global_load_b128 v[248:251], v2, s[38:39] offset:16
	.Lrx_b6_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x16
	v_bfe_u32 v36, v28, 0, 4
	v_bfe_u32 v37, v28, 4, 4
	v_bfe_u32 v38, v28, 8, 4
	v_bfe_u32 v39, v28, 12, 4
	v_bfe_u32 v40, v28, 16, 4
	v_bfe_u32 v41, v28, 20, 4
	v_bfe_u32 v42, v28, 24, 4
	v_bfe_u32 v43, v28, 28, 4
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_cvt_f32_ubyte0_e32 v42, v42
	v_cvt_f32_ubyte0_e32 v43, v43
	v_fma_mix_f32 v36, v16, v36, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v16, v37, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v16, v38, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v16, v39, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v46, v29, 0, 4
	v_bfe_u32 v47, v29, 4, 4
	v_bfe_u32 v48, v29, 8, 4
	v_bfe_u32 v49, v29, 12, 4
	v_bfe_u32 v50, v29, 16, 4
	v_bfe_u32 v51, v29, 20, 4
	v_bfe_u32 v52, v29, 24, 4
	v_bfe_u32 v53, v29, 28, 4
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_cvt_f32_ubyte0_e32 v52, v52
	v_cvt_f32_ubyte0_e32 v53, v53
	v_fma_mix_f32 v46, v18, v46, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v18, v47, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v18, v48, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v18, v49, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v18, v50, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v18, v51, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v18, v52, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v18, v53, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v56, v30, 0, 4
	v_bfe_u32 v57, v30, 4, 4
	v_bfe_u32 v58, v30, 8, 4
	v_bfe_u32 v59, v30, 12, 4
	v_bfe_u32 v60, v30, 16, 4
	v_bfe_u32 v61, v30, 20, 4
	v_bfe_u32 v62, v30, 24, 4
	v_bfe_u32 v63, v30, 28, 4
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f32_ubyte0_e32 v62, v62
	v_cvt_f32_ubyte0_e32 v63, v63
	v_fma_mix_f32 v56, v20, v56, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v20, v57, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v20, v58, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v20, v59, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v20, v60, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v20, v61, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v20, v62, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v20, v63, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v66, v31, 0, 4
	v_bfe_u32 v67, v31, 4, 4
	v_bfe_u32 v68, v31, 8, 4
	v_bfe_u32 v69, v31, 12, 4
	v_bfe_u32 v70, v31, 16, 4
	v_bfe_u32 v71, v31, 20, 4
	v_bfe_u32 v72, v31, 24, 4
	v_bfe_u32 v73, v31, 28, 4
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_cvt_f32_ubyte0_e32 v72, v72
	v_cvt_f32_ubyte0_e32 v73, v73
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v22, v68, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v22, v69, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v22, v70, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v22, v71, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v22, v72, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v22, v73, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v76, v32, 0, 4
	v_bfe_u32 v77, v32, 4, 4
	v_bfe_u32 v78, v32, 8, 4
	v_bfe_u32 v79, v32, 12, 4
	v_bfe_u32 v80, v32, 16, 4
	v_bfe_u32 v81, v32, 20, 4
	v_bfe_u32 v82, v32, 24, 4
	v_bfe_u32 v83, v32, 28, 4
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_cvt_f32_ubyte0_e32 v82, v82
	v_cvt_f32_ubyte0_e32 v83, v83
	v_fma_mix_f32 v76, v24, v76, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v24, v77, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v24, v78, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v24, v79, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v24, v80, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v24, v81, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v82, v24, v82, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v83, v24, v83, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v86, v33, 0, 4
	v_bfe_u32 v87, v33, 4, 4
	v_bfe_u32 v88, v33, 8, 4
	v_bfe_u32 v89, v33, 12, 4
	v_bfe_u32 v90, v33, 16, 4
	v_bfe_u32 v91, v33, 20, 4
	v_bfe_u32 v92, v33, 24, 4
	v_bfe_u32 v93, v33, 28, 4
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_cvt_f32_ubyte0_e32 v92, v92
	v_cvt_f32_ubyte0_e32 v93, v93
	v_fma_mix_f32 v86, v26, v86, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v26, v87, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v26, v88, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v26, v89, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v26, v90, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v26, v91, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v92, v26, v92, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v93, v26, v93, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v44, v36, v204 :: v_dual_mul_f32 v45, v47, v205
	v_dual_mul_f32 v54, v56, v204 :: v_dual_mul_f32 v55, v67, v205
	v_dual_mul_f32 v64, v76, v204 :: v_dual_mul_f32 v65, v87, v205
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v74, v36, v212 :: v_dual_mul_f32 v75, v47, v213
	v_dual_mul_f32 v84, v56, v212 :: v_dual_mul_f32 v85, v67, v213
	v_dual_mul_f32 v94, v76, v212 :: v_dual_mul_f32 v95, v87, v213
	v_dual_fmac_f32 v44, v37, v205 :: v_dual_fmac_f32 v45, v46, v204
	v_dual_fmac_f32 v54, v57, v205 :: v_dual_fmac_f32 v55, v66, v204
	v_dual_fmac_f32 v64, v77, v205 :: v_dual_fmac_f32 v65, v86, v204
	v_dual_fmac_f32 v74, v37, v213 :: v_dual_fmac_f32 v75, v46, v212
	v_dual_fmac_f32 v84, v57, v213 :: v_dual_fmac_f32 v85, v66, v212
	v_dual_fmac_f32 v94, v77, v213 :: v_dual_fmac_f32 v95, v86, v212
	v_dual_fmac_f32 v44, v38, v206 :: v_dual_fmac_f32 v45, v48, v206
	v_dual_fmac_f32 v54, v58, v206 :: v_dual_fmac_f32 v55, v68, v206
	v_dual_fmac_f32 v64, v78, v206 :: v_dual_fmac_f32 v65, v88, v206
	v_dual_fmac_f32 v74, v38, v214 :: v_dual_fmac_f32 v75, v48, v214
	v_dual_fmac_f32 v84, v58, v214 :: v_dual_fmac_f32 v85, v68, v214
	v_dual_fmac_f32 v94, v78, v214 :: v_dual_fmac_f32 v95, v88, v214
	v_dual_fmac_f32 v44, v39, v207 :: v_dual_fmac_f32 v45, v49, v207
	v_dual_fmac_f32 v54, v59, v207 :: v_dual_fmac_f32 v55, v69, v207
	v_dual_fmac_f32 v64, v79, v207 :: v_dual_fmac_f32 v65, v89, v207
	v_dual_fmac_f32 v74, v39, v215 :: v_dual_fmac_f32 v75, v49, v215
	v_dual_fmac_f32 v84, v59, v215 :: v_dual_fmac_f32 v85, v69, v215
	v_dual_fmac_f32 v94, v79, v215 :: v_dual_fmac_f32 v95, v89, v215
	v_dual_fmac_f32 v44, v40, v208 :: v_dual_fmac_f32 v45, v50, v208
	v_dual_fmac_f32 v54, v60, v208 :: v_dual_fmac_f32 v55, v70, v208
	v_dual_fmac_f32 v64, v80, v208 :: v_dual_fmac_f32 v65, v90, v208
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v74, v40, v216 :: v_dual_fmac_f32 v75, v50, v216
	v_dual_fmac_f32 v84, v60, v216 :: v_dual_fmac_f32 v85, v70, v216
	v_dual_fmac_f32 v94, v80, v216 :: v_dual_fmac_f32 v95, v90, v216
	v_dual_fmac_f32 v44, v41, v209 :: v_dual_fmac_f32 v45, v51, v209
	v_dual_fmac_f32 v54, v61, v209 :: v_dual_fmac_f32 v55, v71, v209
	v_dual_fmac_f32 v64, v81, v209 :: v_dual_fmac_f32 v65, v91, v209
	v_dual_fmac_f32 v74, v41, v217 :: v_dual_fmac_f32 v75, v51, v217
	v_dual_fmac_f32 v84, v61, v217 :: v_dual_fmac_f32 v85, v71, v217
	v_dual_fmac_f32 v94, v81, v217 :: v_dual_fmac_f32 v95, v91, v217
	v_dual_fmac_f32 v44, v42, v210 :: v_dual_fmac_f32 v45, v52, v210
	v_dual_fmac_f32 v54, v62, v210 :: v_dual_fmac_f32 v55, v72, v210
	v_dual_fmac_f32 v64, v82, v210 :: v_dual_fmac_f32 v65, v92, v210
	v_dual_fmac_f32 v74, v42, v218 :: v_dual_fmac_f32 v75, v52, v218
	v_dual_fmac_f32 v84, v62, v218 :: v_dual_fmac_f32 v85, v72, v218
	v_dual_fmac_f32 v94, v82, v218 :: v_dual_fmac_f32 v95, v92, v218
	v_dual_fmac_f32 v44, v43, v211 :: v_dual_fmac_f32 v45, v53, v211
	v_dual_fmac_f32 v54, v63, v211 :: v_dual_fmac_f32 v55, v73, v211
	v_dual_fmac_f32 v64, v83, v211 :: v_dual_fmac_f32 v65, v93, v211
	v_dual_fmac_f32 v74, v43, v219 :: v_dual_fmac_f32 v75, v53, v219
	v_dual_fmac_f32 v84, v63, v219 :: v_dual_fmac_f32 v85, v73, v219
	v_dual_fmac_f32 v94, v83, v219 :: v_dual_fmac_f32 v95, v93, v219
	global_load_b128 v[204:207], v3, s[28:29]
	global_load_b128 v[208:211], v3, s[28:29] offset:16
	global_load_b128 v[212:215], v3, s[30:31]
	global_load_b128 v[216:219], v3, s[30:31] offset:16
	v_dual_add_f32 v96, v44, v96 :: v_dual_add_f32 v97, v97, v45
	v_dual_add_f32 v98, v54, v98 :: v_dual_add_f32 v99, v99, v55
	v_dual_add_f32 v100, v64, v100 :: v_dual_add_f32 v101, v101, v65
	v_dual_add_f32 v114, v74, v114 :: v_dual_add_f32 v115, v115, v75
	v_dual_add_f32 v116, v84, v116 :: v_dual_add_f32 v117, v117, v85
	v_dual_add_f32 v118, v94, v118 :: v_dual_add_f32 v119, v119, v95
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v44, v36, v220 :: v_dual_mul_f32 v45, v47, v221
	v_dual_mul_f32 v54, v56, v220 :: v_dual_mul_f32 v55, v67, v221
	v_dual_mul_f32 v64, v76, v220 :: v_dual_mul_f32 v65, v87, v221
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v74, v36, v228 :: v_dual_mul_f32 v75, v47, v229
	v_dual_mul_f32 v84, v56, v228 :: v_dual_mul_f32 v85, v67, v229
	v_dual_mul_f32 v94, v76, v228 :: v_dual_mul_f32 v95, v87, v229
	v_dual_fmac_f32 v44, v37, v221 :: v_dual_fmac_f32 v45, v46, v220
	v_dual_fmac_f32 v54, v57, v221 :: v_dual_fmac_f32 v55, v66, v220
	v_dual_fmac_f32 v64, v77, v221 :: v_dual_fmac_f32 v65, v86, v220
	v_dual_fmac_f32 v74, v37, v229 :: v_dual_fmac_f32 v75, v46, v228
	v_dual_fmac_f32 v84, v57, v229 :: v_dual_fmac_f32 v85, v66, v228
	v_dual_fmac_f32 v94, v77, v229 :: v_dual_fmac_f32 v95, v86, v228
	v_dual_fmac_f32 v44, v38, v222 :: v_dual_fmac_f32 v45, v48, v222
	v_dual_fmac_f32 v54, v58, v222 :: v_dual_fmac_f32 v55, v68, v222
	v_dual_fmac_f32 v64, v78, v222 :: v_dual_fmac_f32 v65, v88, v222
	v_dual_fmac_f32 v74, v38, v230 :: v_dual_fmac_f32 v75, v48, v230
	v_dual_fmac_f32 v84, v58, v230 :: v_dual_fmac_f32 v85, v68, v230
	v_dual_fmac_f32 v94, v78, v230 :: v_dual_fmac_f32 v95, v88, v230
	v_dual_fmac_f32 v44, v39, v223 :: v_dual_fmac_f32 v45, v49, v223
	v_dual_fmac_f32 v54, v59, v223 :: v_dual_fmac_f32 v55, v69, v223
	v_dual_fmac_f32 v64, v79, v223 :: v_dual_fmac_f32 v65, v89, v223
	v_dual_fmac_f32 v74, v39, v231 :: v_dual_fmac_f32 v75, v49, v231
	v_dual_fmac_f32 v84, v59, v231 :: v_dual_fmac_f32 v85, v69, v231
	v_dual_fmac_f32 v94, v79, v231 :: v_dual_fmac_f32 v95, v89, v231
	v_dual_fmac_f32 v44, v40, v224 :: v_dual_fmac_f32 v45, v50, v224
	v_dual_fmac_f32 v54, v60, v224 :: v_dual_fmac_f32 v55, v70, v224
	v_dual_fmac_f32 v64, v80, v224 :: v_dual_fmac_f32 v65, v90, v224
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v74, v40, v232 :: v_dual_fmac_f32 v75, v50, v232
	v_dual_fmac_f32 v84, v60, v232 :: v_dual_fmac_f32 v85, v70, v232
	v_dual_fmac_f32 v94, v80, v232 :: v_dual_fmac_f32 v95, v90, v232
	v_dual_fmac_f32 v44, v41, v225 :: v_dual_fmac_f32 v45, v51, v225
	v_dual_fmac_f32 v54, v61, v225 :: v_dual_fmac_f32 v55, v71, v225
	v_dual_fmac_f32 v64, v81, v225 :: v_dual_fmac_f32 v65, v91, v225
	v_dual_fmac_f32 v74, v41, v233 :: v_dual_fmac_f32 v75, v51, v233
	v_dual_fmac_f32 v84, v61, v233 :: v_dual_fmac_f32 v85, v71, v233
	v_dual_fmac_f32 v94, v81, v233 :: v_dual_fmac_f32 v95, v91, v233
	v_dual_fmac_f32 v44, v42, v226 :: v_dual_fmac_f32 v45, v52, v226
	v_dual_fmac_f32 v54, v62, v226 :: v_dual_fmac_f32 v55, v72, v226
	v_dual_fmac_f32 v64, v82, v226 :: v_dual_fmac_f32 v65, v92, v226
	v_dual_fmac_f32 v74, v42, v234 :: v_dual_fmac_f32 v75, v52, v234
	v_dual_fmac_f32 v84, v62, v234 :: v_dual_fmac_f32 v85, v72, v234
	v_dual_fmac_f32 v94, v82, v234 :: v_dual_fmac_f32 v95, v92, v234
	v_dual_fmac_f32 v44, v43, v227 :: v_dual_fmac_f32 v45, v53, v227
	v_dual_fmac_f32 v54, v63, v227 :: v_dual_fmac_f32 v55, v73, v227
	v_dual_fmac_f32 v64, v83, v227 :: v_dual_fmac_f32 v65, v93, v227
	v_dual_fmac_f32 v74, v43, v235 :: v_dual_fmac_f32 v75, v53, v235
	v_dual_fmac_f32 v84, v63, v235 :: v_dual_fmac_f32 v85, v73, v235
	v_dual_fmac_f32 v94, v83, v235 :: v_dual_fmac_f32 v95, v93, v235
	global_load_b128 v[220:223], v3, s[32:33]
	global_load_b128 v[224:227], v3, s[32:33] offset:16
	global_load_b128 v[228:231], v3, s[34:35]
	global_load_b128 v[232:235], v3, s[34:35] offset:16
	v_dual_add_f32 v132, v44, v132 :: v_dual_add_f32 v133, v133, v45
	v_dual_add_f32 v134, v54, v134 :: v_dual_add_f32 v135, v135, v55
	v_dual_add_f32 v136, v64, v136 :: v_dual_add_f32 v137, v137, v65
	v_dual_add_f32 v150, v74, v150 :: v_dual_add_f32 v151, v151, v75
	v_dual_add_f32 v152, v84, v152 :: v_dual_add_f32 v153, v153, v85
	v_dual_add_f32 v154, v94, v154 :: v_dual_add_f32 v155, v155, v95
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v44, v36, v236 :: v_dual_mul_f32 v45, v47, v237
	v_dual_mul_f32 v54, v56, v236 :: v_dual_mul_f32 v55, v67, v237
	v_dual_mul_f32 v64, v76, v236 :: v_dual_mul_f32 v65, v87, v237
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v74, v36, v244 :: v_dual_mul_f32 v75, v47, v245
	v_dual_mul_f32 v84, v56, v244 :: v_dual_mul_f32 v85, v67, v245
	v_dual_mul_f32 v94, v76, v244 :: v_dual_mul_f32 v95, v87, v245
	v_dual_fmac_f32 v44, v37, v237 :: v_dual_fmac_f32 v45, v46, v236
	v_dual_fmac_f32 v54, v57, v237 :: v_dual_fmac_f32 v55, v66, v236
	v_dual_fmac_f32 v64, v77, v237 :: v_dual_fmac_f32 v65, v86, v236
	v_dual_fmac_f32 v74, v37, v245 :: v_dual_fmac_f32 v75, v46, v244
	v_dual_fmac_f32 v84, v57, v245 :: v_dual_fmac_f32 v85, v66, v244
	v_dual_fmac_f32 v94, v77, v245 :: v_dual_fmac_f32 v95, v86, v244
	v_dual_fmac_f32 v44, v38, v238 :: v_dual_fmac_f32 v45, v48, v238
	v_dual_fmac_f32 v54, v58, v238 :: v_dual_fmac_f32 v55, v68, v238
	v_dual_fmac_f32 v64, v78, v238 :: v_dual_fmac_f32 v65, v88, v238
	v_dual_fmac_f32 v74, v38, v246 :: v_dual_fmac_f32 v75, v48, v246
	v_dual_fmac_f32 v84, v58, v246 :: v_dual_fmac_f32 v85, v68, v246
	v_dual_fmac_f32 v94, v78, v246 :: v_dual_fmac_f32 v95, v88, v246
	v_dual_fmac_f32 v44, v39, v239 :: v_dual_fmac_f32 v45, v49, v239
	v_dual_fmac_f32 v54, v59, v239 :: v_dual_fmac_f32 v55, v69, v239
	v_dual_fmac_f32 v64, v79, v239 :: v_dual_fmac_f32 v65, v89, v239
	v_dual_fmac_f32 v74, v39, v247 :: v_dual_fmac_f32 v75, v49, v247
	v_dual_fmac_f32 v84, v59, v247 :: v_dual_fmac_f32 v85, v69, v247
	v_dual_fmac_f32 v94, v79, v247 :: v_dual_fmac_f32 v95, v89, v247
	v_dual_fmac_f32 v44, v40, v240 :: v_dual_fmac_f32 v45, v50, v240
	v_dual_fmac_f32 v54, v60, v240 :: v_dual_fmac_f32 v55, v70, v240
	v_dual_fmac_f32 v64, v80, v240 :: v_dual_fmac_f32 v65, v90, v240
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v74, v40, v248 :: v_dual_fmac_f32 v75, v50, v248
	v_dual_fmac_f32 v84, v60, v248 :: v_dual_fmac_f32 v85, v70, v248
	v_dual_fmac_f32 v94, v80, v248 :: v_dual_fmac_f32 v95, v90, v248
	v_dual_fmac_f32 v44, v41, v241 :: v_dual_fmac_f32 v45, v51, v241
	v_dual_fmac_f32 v54, v61, v241 :: v_dual_fmac_f32 v55, v71, v241
	v_dual_fmac_f32 v64, v81, v241 :: v_dual_fmac_f32 v65, v91, v241
	v_dual_fmac_f32 v74, v41, v249 :: v_dual_fmac_f32 v75, v51, v249
	v_dual_fmac_f32 v84, v61, v249 :: v_dual_fmac_f32 v85, v71, v249
	v_dual_fmac_f32 v94, v81, v249 :: v_dual_fmac_f32 v95, v91, v249
	v_dual_fmac_f32 v44, v42, v242 :: v_dual_fmac_f32 v45, v52, v242
	v_dual_fmac_f32 v54, v62, v242 :: v_dual_fmac_f32 v55, v72, v242
	v_dual_fmac_f32 v64, v82, v242 :: v_dual_fmac_f32 v65, v92, v242
	v_dual_fmac_f32 v74, v42, v250 :: v_dual_fmac_f32 v75, v52, v250
	v_dual_fmac_f32 v84, v62, v250 :: v_dual_fmac_f32 v85, v72, v250
	v_dual_fmac_f32 v94, v82, v250 :: v_dual_fmac_f32 v95, v92, v250
	v_dual_fmac_f32 v44, v43, v243 :: v_dual_fmac_f32 v45, v53, v243
	v_dual_fmac_f32 v54, v63, v243 :: v_dual_fmac_f32 v55, v73, v243
	v_dual_fmac_f32 v64, v83, v243 :: v_dual_fmac_f32 v65, v93, v243
	v_dual_fmac_f32 v74, v43, v251 :: v_dual_fmac_f32 v75, v53, v251
	v_dual_fmac_f32 v84, v63, v251 :: v_dual_fmac_f32 v85, v73, v251
	v_dual_fmac_f32 v94, v83, v251 :: v_dual_fmac_f32 v95, v93, v251
	global_load_b128 v[236:239], v3, s[36:37]
	global_load_b128 v[240:243], v3, s[36:37] offset:16
	global_load_b128 v[244:247], v3, s[38:39]
	global_load_b128 v[248:251], v3, s[38:39] offset:16
	v_dual_add_f32 v168, v44, v168 :: v_dual_add_f32 v169, v169, v45
	v_dual_add_f32 v170, v54, v170 :: v_dual_add_f32 v171, v171, v55
	v_dual_add_f32 v172, v64, v172 :: v_dual_add_f32 v173, v173, v65
	v_dual_add_f32 v186, v74, v186 :: v_dual_add_f32 v187, v187, v75
	v_dual_add_f32 v188, v84, v188 :: v_dual_add_f32 v189, v189, v85
	v_dual_add_f32 v190, v94, v190 :: v_dual_add_f32 v191, v191, v95
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s0
	s_wait_loadcnt 0x0
	.Lrx_b6_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[204:207], v2, s[28:29]
	global_load_b128 v[208:211], v2, s[28:29] offset:16
	global_load_b128 v[212:215], v2, s[30:31]
	global_load_b128 v[216:219], v2, s[30:31] offset:16
	global_load_b128 v[220:223], v2, s[32:33]
	global_load_b128 v[224:227], v2, s[32:33] offset:16
	global_load_b128 v[228:231], v2, s[34:35]
	global_load_b128 v[232:235], v2, s[34:35] offset:16
	global_load_b128 v[236:239], v2, s[36:37]
	global_load_b128 v[240:243], v2, s[36:37] offset:16
	global_load_b128 v[244:247], v2, s[38:39]
	global_load_b128 v[248:251], v2, s[38:39] offset:16
	.Lrx_b6_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x16
	v_bfe_u32 v36, v28, 0, 4
	v_bfe_u32 v37, v28, 4, 4
	v_bfe_u32 v38, v28, 8, 4
	v_bfe_u32 v39, v28, 12, 4
	v_bfe_u32 v40, v28, 16, 4
	v_bfe_u32 v41, v28, 20, 4
	v_bfe_u32 v42, v28, 24, 4
	v_bfe_u32 v43, v28, 28, 4
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_cvt_f32_ubyte0_e32 v42, v42
	v_cvt_f32_ubyte0_e32 v43, v43
	v_fma_mix_f32 v36, v16, v36, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v16, v37, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v16, v38, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v16, v39, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v46, v29, 0, 4
	v_bfe_u32 v47, v29, 4, 4
	v_bfe_u32 v48, v29, 8, 4
	v_bfe_u32 v49, v29, 12, 4
	v_bfe_u32 v50, v29, 16, 4
	v_bfe_u32 v51, v29, 20, 4
	v_bfe_u32 v52, v29, 24, 4
	v_bfe_u32 v53, v29, 28, 4
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_cvt_f32_ubyte0_e32 v52, v52
	v_cvt_f32_ubyte0_e32 v53, v53
	v_fma_mix_f32 v46, v18, v46, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v18, v47, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v18, v48, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v18, v49, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v18, v50, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v18, v51, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v18, v52, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v18, v53, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v56, v30, 0, 4
	v_bfe_u32 v57, v30, 4, 4
	v_bfe_u32 v58, v30, 8, 4
	v_bfe_u32 v59, v30, 12, 4
	v_bfe_u32 v60, v30, 16, 4
	v_bfe_u32 v61, v30, 20, 4
	v_bfe_u32 v62, v30, 24, 4
	v_bfe_u32 v63, v30, 28, 4
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f32_ubyte0_e32 v62, v62
	v_cvt_f32_ubyte0_e32 v63, v63
	v_fma_mix_f32 v56, v20, v56, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v20, v57, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v20, v58, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v20, v59, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v20, v60, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v20, v61, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v20, v62, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v20, v63, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v66, v31, 0, 4
	v_bfe_u32 v67, v31, 4, 4
	v_bfe_u32 v68, v31, 8, 4
	v_bfe_u32 v69, v31, 12, 4
	v_bfe_u32 v70, v31, 16, 4
	v_bfe_u32 v71, v31, 20, 4
	v_bfe_u32 v72, v31, 24, 4
	v_bfe_u32 v73, v31, 28, 4
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_cvt_f32_ubyte0_e32 v72, v72
	v_cvt_f32_ubyte0_e32 v73, v73
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v22, v68, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v22, v69, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v22, v70, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v22, v71, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v22, v72, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v22, v73, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v76, v32, 0, 4
	v_bfe_u32 v77, v32, 4, 4
	v_bfe_u32 v78, v32, 8, 4
	v_bfe_u32 v79, v32, 12, 4
	v_bfe_u32 v80, v32, 16, 4
	v_bfe_u32 v81, v32, 20, 4
	v_bfe_u32 v82, v32, 24, 4
	v_bfe_u32 v83, v32, 28, 4
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_cvt_f32_ubyte0_e32 v82, v82
	v_cvt_f32_ubyte0_e32 v83, v83
	v_fma_mix_f32 v76, v24, v76, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v24, v77, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v24, v78, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v24, v79, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v24, v80, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v24, v81, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v82, v24, v82, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v83, v24, v83, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v86, v33, 0, 4
	v_bfe_u32 v87, v33, 4, 4
	v_bfe_u32 v88, v33, 8, 4
	v_bfe_u32 v89, v33, 12, 4
	v_bfe_u32 v90, v33, 16, 4
	v_bfe_u32 v91, v33, 20, 4
	v_bfe_u32 v92, v33, 24, 4
	v_bfe_u32 v93, v33, 28, 4
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_cvt_f32_ubyte0_e32 v92, v92
	v_cvt_f32_ubyte0_e32 v93, v93
	v_fma_mix_f32 v86, v26, v86, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v26, v87, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v26, v88, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v26, v89, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v26, v90, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v26, v91, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v92, v26, v92, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v93, v26, v93, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v44, v36, v204 :: v_dual_mul_f32 v45, v47, v205
	v_dual_mul_f32 v54, v56, v204 :: v_dual_mul_f32 v55, v67, v205
	v_dual_mul_f32 v64, v76, v204 :: v_dual_mul_f32 v65, v87, v205
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v74, v36, v212 :: v_dual_mul_f32 v75, v47, v213
	v_dual_mul_f32 v84, v56, v212 :: v_dual_mul_f32 v85, v67, v213
	v_dual_mul_f32 v94, v76, v212 :: v_dual_mul_f32 v95, v87, v213
	v_dual_fmac_f32 v44, v37, v205 :: v_dual_fmac_f32 v45, v46, v204
	v_dual_fmac_f32 v54, v57, v205 :: v_dual_fmac_f32 v55, v66, v204
	v_dual_fmac_f32 v64, v77, v205 :: v_dual_fmac_f32 v65, v86, v204
	v_dual_fmac_f32 v74, v37, v213 :: v_dual_fmac_f32 v75, v46, v212
	v_dual_fmac_f32 v84, v57, v213 :: v_dual_fmac_f32 v85, v66, v212
	v_dual_fmac_f32 v94, v77, v213 :: v_dual_fmac_f32 v95, v86, v212
	v_dual_fmac_f32 v44, v38, v206 :: v_dual_fmac_f32 v45, v48, v206
	v_dual_fmac_f32 v54, v58, v206 :: v_dual_fmac_f32 v55, v68, v206
	v_dual_fmac_f32 v64, v78, v206 :: v_dual_fmac_f32 v65, v88, v206
	v_dual_fmac_f32 v74, v38, v214 :: v_dual_fmac_f32 v75, v48, v214
	v_dual_fmac_f32 v84, v58, v214 :: v_dual_fmac_f32 v85, v68, v214
	v_dual_fmac_f32 v94, v78, v214 :: v_dual_fmac_f32 v95, v88, v214
	v_dual_fmac_f32 v44, v39, v207 :: v_dual_fmac_f32 v45, v49, v207
	v_dual_fmac_f32 v54, v59, v207 :: v_dual_fmac_f32 v55, v69, v207
	v_dual_fmac_f32 v64, v79, v207 :: v_dual_fmac_f32 v65, v89, v207
	v_dual_fmac_f32 v74, v39, v215 :: v_dual_fmac_f32 v75, v49, v215
	v_dual_fmac_f32 v84, v59, v215 :: v_dual_fmac_f32 v85, v69, v215
	v_dual_fmac_f32 v94, v79, v215 :: v_dual_fmac_f32 v95, v89, v215
	v_dual_fmac_f32 v44, v40, v208 :: v_dual_fmac_f32 v45, v50, v208
	v_dual_fmac_f32 v54, v60, v208 :: v_dual_fmac_f32 v55, v70, v208
	v_dual_fmac_f32 v64, v80, v208 :: v_dual_fmac_f32 v65, v90, v208
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v74, v40, v216 :: v_dual_fmac_f32 v75, v50, v216
	v_dual_fmac_f32 v84, v60, v216 :: v_dual_fmac_f32 v85, v70, v216
	v_dual_fmac_f32 v94, v80, v216 :: v_dual_fmac_f32 v95, v90, v216
	v_dual_fmac_f32 v44, v41, v209 :: v_dual_fmac_f32 v45, v51, v209
	v_dual_fmac_f32 v54, v61, v209 :: v_dual_fmac_f32 v55, v71, v209
	v_dual_fmac_f32 v64, v81, v209 :: v_dual_fmac_f32 v65, v91, v209
	v_dual_fmac_f32 v74, v41, v217 :: v_dual_fmac_f32 v75, v51, v217
	v_dual_fmac_f32 v84, v61, v217 :: v_dual_fmac_f32 v85, v71, v217
	v_dual_fmac_f32 v94, v81, v217 :: v_dual_fmac_f32 v95, v91, v217
	v_dual_fmac_f32 v44, v42, v210 :: v_dual_fmac_f32 v45, v52, v210
	v_dual_fmac_f32 v54, v62, v210 :: v_dual_fmac_f32 v55, v72, v210
	v_dual_fmac_f32 v64, v82, v210 :: v_dual_fmac_f32 v65, v92, v210
	v_dual_fmac_f32 v74, v42, v218 :: v_dual_fmac_f32 v75, v52, v218
	v_dual_fmac_f32 v84, v62, v218 :: v_dual_fmac_f32 v85, v72, v218
	v_dual_fmac_f32 v94, v82, v218 :: v_dual_fmac_f32 v95, v92, v218
	v_dual_fmac_f32 v44, v43, v211 :: v_dual_fmac_f32 v45, v53, v211
	v_dual_fmac_f32 v54, v63, v211 :: v_dual_fmac_f32 v55, v73, v211
	v_dual_fmac_f32 v64, v83, v211 :: v_dual_fmac_f32 v65, v93, v211
	v_dual_fmac_f32 v74, v43, v219 :: v_dual_fmac_f32 v75, v53, v219
	v_dual_fmac_f32 v84, v63, v219 :: v_dual_fmac_f32 v85, v73, v219
	v_dual_fmac_f32 v94, v83, v219 :: v_dual_fmac_f32 v95, v93, v219
	global_load_b128 v[204:207], v3, s[28:29]
	global_load_b128 v[208:211], v3, s[28:29] offset:16
	global_load_b128 v[212:215], v3, s[30:31]
	global_load_b128 v[216:219], v3, s[30:31] offset:16
	v_dual_add_f32 v102, v44, v102 :: v_dual_add_f32 v103, v103, v45
	v_dual_add_f32 v104, v54, v104 :: v_dual_add_f32 v105, v105, v55
	v_dual_add_f32 v106, v64, v106 :: v_dual_add_f32 v107, v107, v65
	v_dual_add_f32 v120, v74, v120 :: v_dual_add_f32 v121, v121, v75
	v_dual_add_f32 v122, v84, v122 :: v_dual_add_f32 v123, v123, v85
	v_dual_add_f32 v124, v94, v124 :: v_dual_add_f32 v125, v125, v95
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v44, v36, v220 :: v_dual_mul_f32 v45, v47, v221
	v_dual_mul_f32 v54, v56, v220 :: v_dual_mul_f32 v55, v67, v221
	v_dual_mul_f32 v64, v76, v220 :: v_dual_mul_f32 v65, v87, v221
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v74, v36, v228 :: v_dual_mul_f32 v75, v47, v229
	v_dual_mul_f32 v84, v56, v228 :: v_dual_mul_f32 v85, v67, v229
	v_dual_mul_f32 v94, v76, v228 :: v_dual_mul_f32 v95, v87, v229
	v_dual_fmac_f32 v44, v37, v221 :: v_dual_fmac_f32 v45, v46, v220
	v_dual_fmac_f32 v54, v57, v221 :: v_dual_fmac_f32 v55, v66, v220
	v_dual_fmac_f32 v64, v77, v221 :: v_dual_fmac_f32 v65, v86, v220
	v_dual_fmac_f32 v74, v37, v229 :: v_dual_fmac_f32 v75, v46, v228
	v_dual_fmac_f32 v84, v57, v229 :: v_dual_fmac_f32 v85, v66, v228
	v_dual_fmac_f32 v94, v77, v229 :: v_dual_fmac_f32 v95, v86, v228
	v_dual_fmac_f32 v44, v38, v222 :: v_dual_fmac_f32 v45, v48, v222
	v_dual_fmac_f32 v54, v58, v222 :: v_dual_fmac_f32 v55, v68, v222
	v_dual_fmac_f32 v64, v78, v222 :: v_dual_fmac_f32 v65, v88, v222
	v_dual_fmac_f32 v74, v38, v230 :: v_dual_fmac_f32 v75, v48, v230
	v_dual_fmac_f32 v84, v58, v230 :: v_dual_fmac_f32 v85, v68, v230
	v_dual_fmac_f32 v94, v78, v230 :: v_dual_fmac_f32 v95, v88, v230
	v_dual_fmac_f32 v44, v39, v223 :: v_dual_fmac_f32 v45, v49, v223
	v_dual_fmac_f32 v54, v59, v223 :: v_dual_fmac_f32 v55, v69, v223
	v_dual_fmac_f32 v64, v79, v223 :: v_dual_fmac_f32 v65, v89, v223
	v_dual_fmac_f32 v74, v39, v231 :: v_dual_fmac_f32 v75, v49, v231
	v_dual_fmac_f32 v84, v59, v231 :: v_dual_fmac_f32 v85, v69, v231
	v_dual_fmac_f32 v94, v79, v231 :: v_dual_fmac_f32 v95, v89, v231
	v_dual_fmac_f32 v44, v40, v224 :: v_dual_fmac_f32 v45, v50, v224
	v_dual_fmac_f32 v54, v60, v224 :: v_dual_fmac_f32 v55, v70, v224
	v_dual_fmac_f32 v64, v80, v224 :: v_dual_fmac_f32 v65, v90, v224
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v74, v40, v232 :: v_dual_fmac_f32 v75, v50, v232
	v_dual_fmac_f32 v84, v60, v232 :: v_dual_fmac_f32 v85, v70, v232
	v_dual_fmac_f32 v94, v80, v232 :: v_dual_fmac_f32 v95, v90, v232
	v_dual_fmac_f32 v44, v41, v225 :: v_dual_fmac_f32 v45, v51, v225
	v_dual_fmac_f32 v54, v61, v225 :: v_dual_fmac_f32 v55, v71, v225
	v_dual_fmac_f32 v64, v81, v225 :: v_dual_fmac_f32 v65, v91, v225
	v_dual_fmac_f32 v74, v41, v233 :: v_dual_fmac_f32 v75, v51, v233
	v_dual_fmac_f32 v84, v61, v233 :: v_dual_fmac_f32 v85, v71, v233
	v_dual_fmac_f32 v94, v81, v233 :: v_dual_fmac_f32 v95, v91, v233
	v_dual_fmac_f32 v44, v42, v226 :: v_dual_fmac_f32 v45, v52, v226
	v_dual_fmac_f32 v54, v62, v226 :: v_dual_fmac_f32 v55, v72, v226
	v_dual_fmac_f32 v64, v82, v226 :: v_dual_fmac_f32 v65, v92, v226
	v_dual_fmac_f32 v74, v42, v234 :: v_dual_fmac_f32 v75, v52, v234
	v_dual_fmac_f32 v84, v62, v234 :: v_dual_fmac_f32 v85, v72, v234
	v_dual_fmac_f32 v94, v82, v234 :: v_dual_fmac_f32 v95, v92, v234
	v_dual_fmac_f32 v44, v43, v227 :: v_dual_fmac_f32 v45, v53, v227
	v_dual_fmac_f32 v54, v63, v227 :: v_dual_fmac_f32 v55, v73, v227
	v_dual_fmac_f32 v64, v83, v227 :: v_dual_fmac_f32 v65, v93, v227
	v_dual_fmac_f32 v74, v43, v235 :: v_dual_fmac_f32 v75, v53, v235
	v_dual_fmac_f32 v84, v63, v235 :: v_dual_fmac_f32 v85, v73, v235
	v_dual_fmac_f32 v94, v83, v235 :: v_dual_fmac_f32 v95, v93, v235
	global_load_b128 v[220:223], v3, s[32:33]
	global_load_b128 v[224:227], v3, s[32:33] offset:16
	global_load_b128 v[228:231], v3, s[34:35]
	global_load_b128 v[232:235], v3, s[34:35] offset:16
	v_dual_add_f32 v138, v44, v138 :: v_dual_add_f32 v139, v139, v45
	v_dual_add_f32 v140, v54, v140 :: v_dual_add_f32 v141, v141, v55
	v_dual_add_f32 v142, v64, v142 :: v_dual_add_f32 v143, v143, v65
	v_dual_add_f32 v156, v74, v156 :: v_dual_add_f32 v157, v157, v75
	v_dual_add_f32 v158, v84, v158 :: v_dual_add_f32 v159, v159, v85
	v_dual_add_f32 v160, v94, v160 :: v_dual_add_f32 v161, v161, v95
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v44, v36, v236 :: v_dual_mul_f32 v45, v47, v237
	v_dual_mul_f32 v54, v56, v236 :: v_dual_mul_f32 v55, v67, v237
	v_dual_mul_f32 v64, v76, v236 :: v_dual_mul_f32 v65, v87, v237
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v74, v36, v244 :: v_dual_mul_f32 v75, v47, v245
	v_dual_mul_f32 v84, v56, v244 :: v_dual_mul_f32 v85, v67, v245
	v_dual_mul_f32 v94, v76, v244 :: v_dual_mul_f32 v95, v87, v245
	v_dual_fmac_f32 v44, v37, v237 :: v_dual_fmac_f32 v45, v46, v236
	v_dual_fmac_f32 v54, v57, v237 :: v_dual_fmac_f32 v55, v66, v236
	v_dual_fmac_f32 v64, v77, v237 :: v_dual_fmac_f32 v65, v86, v236
	v_dual_fmac_f32 v74, v37, v245 :: v_dual_fmac_f32 v75, v46, v244
	v_dual_fmac_f32 v84, v57, v245 :: v_dual_fmac_f32 v85, v66, v244
	v_dual_fmac_f32 v94, v77, v245 :: v_dual_fmac_f32 v95, v86, v244
	v_dual_fmac_f32 v44, v38, v238 :: v_dual_fmac_f32 v45, v48, v238
	v_dual_fmac_f32 v54, v58, v238 :: v_dual_fmac_f32 v55, v68, v238
	v_dual_fmac_f32 v64, v78, v238 :: v_dual_fmac_f32 v65, v88, v238
	v_dual_fmac_f32 v74, v38, v246 :: v_dual_fmac_f32 v75, v48, v246
	v_dual_fmac_f32 v84, v58, v246 :: v_dual_fmac_f32 v85, v68, v246
	v_dual_fmac_f32 v94, v78, v246 :: v_dual_fmac_f32 v95, v88, v246
	v_dual_fmac_f32 v44, v39, v239 :: v_dual_fmac_f32 v45, v49, v239
	v_dual_fmac_f32 v54, v59, v239 :: v_dual_fmac_f32 v55, v69, v239
	v_dual_fmac_f32 v64, v79, v239 :: v_dual_fmac_f32 v65, v89, v239
	v_dual_fmac_f32 v74, v39, v247 :: v_dual_fmac_f32 v75, v49, v247
	v_dual_fmac_f32 v84, v59, v247 :: v_dual_fmac_f32 v85, v69, v247
	v_dual_fmac_f32 v94, v79, v247 :: v_dual_fmac_f32 v95, v89, v247
	v_dual_fmac_f32 v44, v40, v240 :: v_dual_fmac_f32 v45, v50, v240
	v_dual_fmac_f32 v54, v60, v240 :: v_dual_fmac_f32 v55, v70, v240
	v_dual_fmac_f32 v64, v80, v240 :: v_dual_fmac_f32 v65, v90, v240
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v74, v40, v248 :: v_dual_fmac_f32 v75, v50, v248
	v_dual_fmac_f32 v84, v60, v248 :: v_dual_fmac_f32 v85, v70, v248
	v_dual_fmac_f32 v94, v80, v248 :: v_dual_fmac_f32 v95, v90, v248
	v_dual_fmac_f32 v44, v41, v241 :: v_dual_fmac_f32 v45, v51, v241
	v_dual_fmac_f32 v54, v61, v241 :: v_dual_fmac_f32 v55, v71, v241
	v_dual_fmac_f32 v64, v81, v241 :: v_dual_fmac_f32 v65, v91, v241
	v_dual_fmac_f32 v74, v41, v249 :: v_dual_fmac_f32 v75, v51, v249
	v_dual_fmac_f32 v84, v61, v249 :: v_dual_fmac_f32 v85, v71, v249
	v_dual_fmac_f32 v94, v81, v249 :: v_dual_fmac_f32 v95, v91, v249
	v_dual_fmac_f32 v44, v42, v242 :: v_dual_fmac_f32 v45, v52, v242
	v_dual_fmac_f32 v54, v62, v242 :: v_dual_fmac_f32 v55, v72, v242
	v_dual_fmac_f32 v64, v82, v242 :: v_dual_fmac_f32 v65, v92, v242
	v_dual_fmac_f32 v74, v42, v250 :: v_dual_fmac_f32 v75, v52, v250
	v_dual_fmac_f32 v84, v62, v250 :: v_dual_fmac_f32 v85, v72, v250
	v_dual_fmac_f32 v94, v82, v250 :: v_dual_fmac_f32 v95, v92, v250
	v_dual_fmac_f32 v44, v43, v243 :: v_dual_fmac_f32 v45, v53, v243
	v_dual_fmac_f32 v54, v63, v243 :: v_dual_fmac_f32 v55, v73, v243
	v_dual_fmac_f32 v64, v83, v243 :: v_dual_fmac_f32 v65, v93, v243
	v_dual_fmac_f32 v74, v43, v251 :: v_dual_fmac_f32 v75, v53, v251
	v_dual_fmac_f32 v84, v63, v251 :: v_dual_fmac_f32 v85, v73, v251
	v_dual_fmac_f32 v94, v83, v251 :: v_dual_fmac_f32 v95, v93, v251
	global_load_b128 v[236:239], v3, s[36:37]
	global_load_b128 v[240:243], v3, s[36:37] offset:16
	global_load_b128 v[244:247], v3, s[38:39]
	global_load_b128 v[248:251], v3, s[38:39] offset:16
	v_dual_add_f32 v174, v44, v174 :: v_dual_add_f32 v175, v175, v45
	v_dual_add_f32 v176, v54, v176 :: v_dual_add_f32 v177, v177, v55
	v_dual_add_f32 v178, v64, v178 :: v_dual_add_f32 v179, v179, v65
	v_dual_add_f32 v192, v74, v192 :: v_dual_add_f32 v193, v193, v75
	v_dual_add_f32 v194, v84, v194 :: v_dual_add_f32 v195, v195, v85
	v_dual_add_f32 v196, v94, v196 :: v_dual_add_f32 v197, v197, v95
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s1
	s_wait_loadcnt 0x0
	.Lrx_b6_s1_done:
	v_dual_add_f32 v96, v96, v102 :: v_dual_add_f32 v97, v97, v103
	v_dual_add_f32 v98, v98, v104 :: v_dual_add_f32 v99, v99, v105
	v_dual_add_f32 v100, v100, v106 :: v_dual_add_f32 v101, v101, v107
	v_dual_add_f32 v114, v114, v120 :: v_dual_add_f32 v115, v115, v121
	v_dual_add_f32 v116, v116, v122 :: v_dual_add_f32 v117, v117, v123
	v_dual_add_f32 v118, v118, v124 :: v_dual_add_f32 v119, v119, v125
	v_dual_add_f32 v132, v132, v138 :: v_dual_add_f32 v133, v133, v139
	v_dual_add_f32 v134, v134, v140 :: v_dual_add_f32 v135, v135, v141
	v_dual_add_f32 v136, v136, v142 :: v_dual_add_f32 v137, v137, v143
	v_dual_add_f32 v150, v150, v156 :: v_dual_add_f32 v151, v151, v157
	v_dual_add_f32 v152, v152, v158 :: v_dual_add_f32 v153, v153, v159
	v_dual_add_f32 v154, v154, v160 :: v_dual_add_f32 v155, v155, v161
	v_dual_add_f32 v168, v168, v174 :: v_dual_add_f32 v169, v169, v175
	v_dual_add_f32 v170, v170, v176 :: v_dual_add_f32 v171, v171, v177
	v_dual_add_f32 v172, v172, v178 :: v_dual_add_f32 v173, v173, v179
	v_dual_add_f32 v186, v186, v192 :: v_dual_add_f32 v187, v187, v193
	v_dual_add_f32 v188, v188, v194 :: v_dual_add_f32 v189, v189, v195
	v_dual_add_f32 v190, v190, v196 :: v_dual_add_f32 v191, v191, v197
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[204:207], v2, s[28:29]
	global_load_b128 v[208:211], v2, s[28:29] offset:16
	global_load_b128 v[212:215], v2, s[30:31]
	global_load_b128 v[216:219], v2, s[30:31] offset:16
	global_load_b128 v[220:223], v2, s[32:33]
	global_load_b128 v[224:227], v2, s[32:33] offset:16
	global_load_b128 v[228:231], v2, s[34:35]
	global_load_b128 v[232:235], v2, s[34:35] offset:16
	global_load_b128 v[236:239], v2, s[36:37]
	global_load_b128 v[240:243], v2, s[36:37] offset:16
	global_load_b128 v[244:247], v2, s[38:39]
	global_load_b128 v[248:251], v2, s[38:39] offset:16
	.Lrx_b6_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x16
	v_bfe_u32 v36, v28, 0, 4
	v_bfe_u32 v37, v28, 4, 4
	v_bfe_u32 v38, v28, 8, 4
	v_bfe_u32 v39, v28, 12, 4
	v_bfe_u32 v40, v28, 16, 4
	v_bfe_u32 v41, v28, 20, 4
	v_bfe_u32 v42, v28, 24, 4
	v_bfe_u32 v43, v28, 28, 4
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_cvt_f32_ubyte0_e32 v42, v42
	v_cvt_f32_ubyte0_e32 v43, v43
	v_fma_mix_f32 v36, v16, v36, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v16, v37, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v16, v38, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v16, v39, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v46, v29, 0, 4
	v_bfe_u32 v47, v29, 4, 4
	v_bfe_u32 v48, v29, 8, 4
	v_bfe_u32 v49, v29, 12, 4
	v_bfe_u32 v50, v29, 16, 4
	v_bfe_u32 v51, v29, 20, 4
	v_bfe_u32 v52, v29, 24, 4
	v_bfe_u32 v53, v29, 28, 4
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_cvt_f32_ubyte0_e32 v52, v52
	v_cvt_f32_ubyte0_e32 v53, v53
	v_fma_mix_f32 v46, v18, v46, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v18, v47, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v18, v48, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v18, v49, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v18, v50, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v18, v51, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v18, v52, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v18, v53, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v56, v30, 0, 4
	v_bfe_u32 v57, v30, 4, 4
	v_bfe_u32 v58, v30, 8, 4
	v_bfe_u32 v59, v30, 12, 4
	v_bfe_u32 v60, v30, 16, 4
	v_bfe_u32 v61, v30, 20, 4
	v_bfe_u32 v62, v30, 24, 4
	v_bfe_u32 v63, v30, 28, 4
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f32_ubyte0_e32 v62, v62
	v_cvt_f32_ubyte0_e32 v63, v63
	v_fma_mix_f32 v56, v20, v56, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v20, v57, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v20, v58, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v20, v59, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v20, v60, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v20, v61, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v20, v62, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v20, v63, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v66, v31, 0, 4
	v_bfe_u32 v67, v31, 4, 4
	v_bfe_u32 v68, v31, 8, 4
	v_bfe_u32 v69, v31, 12, 4
	v_bfe_u32 v70, v31, 16, 4
	v_bfe_u32 v71, v31, 20, 4
	v_bfe_u32 v72, v31, 24, 4
	v_bfe_u32 v73, v31, 28, 4
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_cvt_f32_ubyte0_e32 v72, v72
	v_cvt_f32_ubyte0_e32 v73, v73
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v22, v68, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v22, v69, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v22, v70, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v22, v71, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v22, v72, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v22, v73, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v76, v32, 0, 4
	v_bfe_u32 v77, v32, 4, 4
	v_bfe_u32 v78, v32, 8, 4
	v_bfe_u32 v79, v32, 12, 4
	v_bfe_u32 v80, v32, 16, 4
	v_bfe_u32 v81, v32, 20, 4
	v_bfe_u32 v82, v32, 24, 4
	v_bfe_u32 v83, v32, 28, 4
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_cvt_f32_ubyte0_e32 v82, v82
	v_cvt_f32_ubyte0_e32 v83, v83
	v_fma_mix_f32 v76, v24, v76, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v24, v77, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v24, v78, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v24, v79, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v24, v80, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v24, v81, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v82, v24, v82, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v83, v24, v83, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v86, v33, 0, 4
	v_bfe_u32 v87, v33, 4, 4
	v_bfe_u32 v88, v33, 8, 4
	v_bfe_u32 v89, v33, 12, 4
	v_bfe_u32 v90, v33, 16, 4
	v_bfe_u32 v91, v33, 20, 4
	v_bfe_u32 v92, v33, 24, 4
	v_bfe_u32 v93, v33, 28, 4
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_cvt_f32_ubyte0_e32 v92, v92
	v_cvt_f32_ubyte0_e32 v93, v93
	v_fma_mix_f32 v86, v26, v86, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v26, v87, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v26, v88, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v26, v89, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v26, v90, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v26, v91, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v92, v26, v92, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v93, v26, v93, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v44, v36, v204 :: v_dual_mul_f32 v45, v47, v205
	v_dual_mul_f32 v54, v56, v204 :: v_dual_mul_f32 v55, v67, v205
	v_dual_mul_f32 v64, v76, v204 :: v_dual_mul_f32 v65, v87, v205
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v74, v36, v212 :: v_dual_mul_f32 v75, v47, v213
	v_dual_mul_f32 v84, v56, v212 :: v_dual_mul_f32 v85, v67, v213
	v_dual_mul_f32 v94, v76, v212 :: v_dual_mul_f32 v95, v87, v213
	v_dual_fmac_f32 v44, v37, v205 :: v_dual_fmac_f32 v45, v46, v204
	v_dual_fmac_f32 v54, v57, v205 :: v_dual_fmac_f32 v55, v66, v204
	v_dual_fmac_f32 v64, v77, v205 :: v_dual_fmac_f32 v65, v86, v204
	v_dual_fmac_f32 v74, v37, v213 :: v_dual_fmac_f32 v75, v46, v212
	v_dual_fmac_f32 v84, v57, v213 :: v_dual_fmac_f32 v85, v66, v212
	v_dual_fmac_f32 v94, v77, v213 :: v_dual_fmac_f32 v95, v86, v212
	v_dual_fmac_f32 v44, v38, v206 :: v_dual_fmac_f32 v45, v48, v206
	v_dual_fmac_f32 v54, v58, v206 :: v_dual_fmac_f32 v55, v68, v206
	v_dual_fmac_f32 v64, v78, v206 :: v_dual_fmac_f32 v65, v88, v206
	v_dual_fmac_f32 v74, v38, v214 :: v_dual_fmac_f32 v75, v48, v214
	v_dual_fmac_f32 v84, v58, v214 :: v_dual_fmac_f32 v85, v68, v214
	v_dual_fmac_f32 v94, v78, v214 :: v_dual_fmac_f32 v95, v88, v214
	v_dual_fmac_f32 v44, v39, v207 :: v_dual_fmac_f32 v45, v49, v207
	v_dual_fmac_f32 v54, v59, v207 :: v_dual_fmac_f32 v55, v69, v207
	v_dual_fmac_f32 v64, v79, v207 :: v_dual_fmac_f32 v65, v89, v207
	v_dual_fmac_f32 v74, v39, v215 :: v_dual_fmac_f32 v75, v49, v215
	v_dual_fmac_f32 v84, v59, v215 :: v_dual_fmac_f32 v85, v69, v215
	v_dual_fmac_f32 v94, v79, v215 :: v_dual_fmac_f32 v95, v89, v215
	v_dual_fmac_f32 v44, v40, v208 :: v_dual_fmac_f32 v45, v50, v208
	v_dual_fmac_f32 v54, v60, v208 :: v_dual_fmac_f32 v55, v70, v208
	v_dual_fmac_f32 v64, v80, v208 :: v_dual_fmac_f32 v65, v90, v208
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v74, v40, v216 :: v_dual_fmac_f32 v75, v50, v216
	v_dual_fmac_f32 v84, v60, v216 :: v_dual_fmac_f32 v85, v70, v216
	v_dual_fmac_f32 v94, v80, v216 :: v_dual_fmac_f32 v95, v90, v216
	v_dual_fmac_f32 v44, v41, v209 :: v_dual_fmac_f32 v45, v51, v209
	v_dual_fmac_f32 v54, v61, v209 :: v_dual_fmac_f32 v55, v71, v209
	v_dual_fmac_f32 v64, v81, v209 :: v_dual_fmac_f32 v65, v91, v209
	v_dual_fmac_f32 v74, v41, v217 :: v_dual_fmac_f32 v75, v51, v217
	v_dual_fmac_f32 v84, v61, v217 :: v_dual_fmac_f32 v85, v71, v217
	v_dual_fmac_f32 v94, v81, v217 :: v_dual_fmac_f32 v95, v91, v217
	v_dual_fmac_f32 v44, v42, v210 :: v_dual_fmac_f32 v45, v52, v210
	v_dual_fmac_f32 v54, v62, v210 :: v_dual_fmac_f32 v55, v72, v210
	v_dual_fmac_f32 v64, v82, v210 :: v_dual_fmac_f32 v65, v92, v210
	v_dual_fmac_f32 v74, v42, v218 :: v_dual_fmac_f32 v75, v52, v218
	v_dual_fmac_f32 v84, v62, v218 :: v_dual_fmac_f32 v85, v72, v218
	v_dual_fmac_f32 v94, v82, v218 :: v_dual_fmac_f32 v95, v92, v218
	v_dual_fmac_f32 v44, v43, v211 :: v_dual_fmac_f32 v45, v53, v211
	v_dual_fmac_f32 v54, v63, v211 :: v_dual_fmac_f32 v55, v73, v211
	v_dual_fmac_f32 v64, v83, v211 :: v_dual_fmac_f32 v65, v93, v211
	v_dual_fmac_f32 v74, v43, v219 :: v_dual_fmac_f32 v75, v53, v219
	v_dual_fmac_f32 v84, v63, v219 :: v_dual_fmac_f32 v85, v73, v219
	v_dual_fmac_f32 v94, v83, v219 :: v_dual_fmac_f32 v95, v93, v219
	global_load_b128 v[204:207], v3, s[28:29]
	global_load_b128 v[208:211], v3, s[28:29] offset:16
	global_load_b128 v[212:215], v3, s[30:31]
	global_load_b128 v[216:219], v3, s[30:31] offset:16
	v_dual_add_f32 v102, v44, v102 :: v_dual_add_f32 v103, v103, v45
	v_dual_add_f32 v104, v54, v104 :: v_dual_add_f32 v105, v105, v55
	v_dual_add_f32 v106, v64, v106 :: v_dual_add_f32 v107, v107, v65
	v_dual_add_f32 v120, v74, v120 :: v_dual_add_f32 v121, v121, v75
	v_dual_add_f32 v122, v84, v122 :: v_dual_add_f32 v123, v123, v85
	v_dual_add_f32 v124, v94, v124 :: v_dual_add_f32 v125, v125, v95
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v44, v36, v220 :: v_dual_mul_f32 v45, v47, v221
	v_dual_mul_f32 v54, v56, v220 :: v_dual_mul_f32 v55, v67, v221
	v_dual_mul_f32 v64, v76, v220 :: v_dual_mul_f32 v65, v87, v221
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v74, v36, v228 :: v_dual_mul_f32 v75, v47, v229
	v_dual_mul_f32 v84, v56, v228 :: v_dual_mul_f32 v85, v67, v229
	v_dual_mul_f32 v94, v76, v228 :: v_dual_mul_f32 v95, v87, v229
	v_dual_fmac_f32 v44, v37, v221 :: v_dual_fmac_f32 v45, v46, v220
	v_dual_fmac_f32 v54, v57, v221 :: v_dual_fmac_f32 v55, v66, v220
	v_dual_fmac_f32 v64, v77, v221 :: v_dual_fmac_f32 v65, v86, v220
	v_dual_fmac_f32 v74, v37, v229 :: v_dual_fmac_f32 v75, v46, v228
	v_dual_fmac_f32 v84, v57, v229 :: v_dual_fmac_f32 v85, v66, v228
	v_dual_fmac_f32 v94, v77, v229 :: v_dual_fmac_f32 v95, v86, v228
	v_dual_fmac_f32 v44, v38, v222 :: v_dual_fmac_f32 v45, v48, v222
	v_dual_fmac_f32 v54, v58, v222 :: v_dual_fmac_f32 v55, v68, v222
	v_dual_fmac_f32 v64, v78, v222 :: v_dual_fmac_f32 v65, v88, v222
	v_dual_fmac_f32 v74, v38, v230 :: v_dual_fmac_f32 v75, v48, v230
	v_dual_fmac_f32 v84, v58, v230 :: v_dual_fmac_f32 v85, v68, v230
	v_dual_fmac_f32 v94, v78, v230 :: v_dual_fmac_f32 v95, v88, v230
	v_dual_fmac_f32 v44, v39, v223 :: v_dual_fmac_f32 v45, v49, v223
	v_dual_fmac_f32 v54, v59, v223 :: v_dual_fmac_f32 v55, v69, v223
	v_dual_fmac_f32 v64, v79, v223 :: v_dual_fmac_f32 v65, v89, v223
	v_dual_fmac_f32 v74, v39, v231 :: v_dual_fmac_f32 v75, v49, v231
	v_dual_fmac_f32 v84, v59, v231 :: v_dual_fmac_f32 v85, v69, v231
	v_dual_fmac_f32 v94, v79, v231 :: v_dual_fmac_f32 v95, v89, v231
	v_dual_fmac_f32 v44, v40, v224 :: v_dual_fmac_f32 v45, v50, v224
	v_dual_fmac_f32 v54, v60, v224 :: v_dual_fmac_f32 v55, v70, v224
	v_dual_fmac_f32 v64, v80, v224 :: v_dual_fmac_f32 v65, v90, v224
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v74, v40, v232 :: v_dual_fmac_f32 v75, v50, v232
	v_dual_fmac_f32 v84, v60, v232 :: v_dual_fmac_f32 v85, v70, v232
	v_dual_fmac_f32 v94, v80, v232 :: v_dual_fmac_f32 v95, v90, v232
	v_dual_fmac_f32 v44, v41, v225 :: v_dual_fmac_f32 v45, v51, v225
	v_dual_fmac_f32 v54, v61, v225 :: v_dual_fmac_f32 v55, v71, v225
	v_dual_fmac_f32 v64, v81, v225 :: v_dual_fmac_f32 v65, v91, v225
	v_dual_fmac_f32 v74, v41, v233 :: v_dual_fmac_f32 v75, v51, v233
	v_dual_fmac_f32 v84, v61, v233 :: v_dual_fmac_f32 v85, v71, v233
	v_dual_fmac_f32 v94, v81, v233 :: v_dual_fmac_f32 v95, v91, v233
	v_dual_fmac_f32 v44, v42, v226 :: v_dual_fmac_f32 v45, v52, v226
	v_dual_fmac_f32 v54, v62, v226 :: v_dual_fmac_f32 v55, v72, v226
	v_dual_fmac_f32 v64, v82, v226 :: v_dual_fmac_f32 v65, v92, v226
	v_dual_fmac_f32 v74, v42, v234 :: v_dual_fmac_f32 v75, v52, v234
	v_dual_fmac_f32 v84, v62, v234 :: v_dual_fmac_f32 v85, v72, v234
	v_dual_fmac_f32 v94, v82, v234 :: v_dual_fmac_f32 v95, v92, v234
	v_dual_fmac_f32 v44, v43, v227 :: v_dual_fmac_f32 v45, v53, v227
	v_dual_fmac_f32 v54, v63, v227 :: v_dual_fmac_f32 v55, v73, v227
	v_dual_fmac_f32 v64, v83, v227 :: v_dual_fmac_f32 v65, v93, v227
	v_dual_fmac_f32 v74, v43, v235 :: v_dual_fmac_f32 v75, v53, v235
	v_dual_fmac_f32 v84, v63, v235 :: v_dual_fmac_f32 v85, v73, v235
	v_dual_fmac_f32 v94, v83, v235 :: v_dual_fmac_f32 v95, v93, v235
	global_load_b128 v[220:223], v3, s[32:33]
	global_load_b128 v[224:227], v3, s[32:33] offset:16
	global_load_b128 v[228:231], v3, s[34:35]
	global_load_b128 v[232:235], v3, s[34:35] offset:16
	v_dual_add_f32 v138, v44, v138 :: v_dual_add_f32 v139, v139, v45
	v_dual_add_f32 v140, v54, v140 :: v_dual_add_f32 v141, v141, v55
	v_dual_add_f32 v142, v64, v142 :: v_dual_add_f32 v143, v143, v65
	v_dual_add_f32 v156, v74, v156 :: v_dual_add_f32 v157, v157, v75
	v_dual_add_f32 v158, v84, v158 :: v_dual_add_f32 v159, v159, v85
	v_dual_add_f32 v160, v94, v160 :: v_dual_add_f32 v161, v161, v95
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v44, v36, v236 :: v_dual_mul_f32 v45, v47, v237
	v_dual_mul_f32 v54, v56, v236 :: v_dual_mul_f32 v55, v67, v237
	v_dual_mul_f32 v64, v76, v236 :: v_dual_mul_f32 v65, v87, v237
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v74, v36, v244 :: v_dual_mul_f32 v75, v47, v245
	v_dual_mul_f32 v84, v56, v244 :: v_dual_mul_f32 v85, v67, v245
	v_dual_mul_f32 v94, v76, v244 :: v_dual_mul_f32 v95, v87, v245
	v_dual_fmac_f32 v44, v37, v237 :: v_dual_fmac_f32 v45, v46, v236
	v_dual_fmac_f32 v54, v57, v237 :: v_dual_fmac_f32 v55, v66, v236
	v_dual_fmac_f32 v64, v77, v237 :: v_dual_fmac_f32 v65, v86, v236
	v_dual_fmac_f32 v74, v37, v245 :: v_dual_fmac_f32 v75, v46, v244
	v_dual_fmac_f32 v84, v57, v245 :: v_dual_fmac_f32 v85, v66, v244
	v_dual_fmac_f32 v94, v77, v245 :: v_dual_fmac_f32 v95, v86, v244
	v_dual_fmac_f32 v44, v38, v238 :: v_dual_fmac_f32 v45, v48, v238
	v_dual_fmac_f32 v54, v58, v238 :: v_dual_fmac_f32 v55, v68, v238
	v_dual_fmac_f32 v64, v78, v238 :: v_dual_fmac_f32 v65, v88, v238
	v_dual_fmac_f32 v74, v38, v246 :: v_dual_fmac_f32 v75, v48, v246
	v_dual_fmac_f32 v84, v58, v246 :: v_dual_fmac_f32 v85, v68, v246
	v_dual_fmac_f32 v94, v78, v246 :: v_dual_fmac_f32 v95, v88, v246
	v_dual_fmac_f32 v44, v39, v239 :: v_dual_fmac_f32 v45, v49, v239
	v_dual_fmac_f32 v54, v59, v239 :: v_dual_fmac_f32 v55, v69, v239
	v_dual_fmac_f32 v64, v79, v239 :: v_dual_fmac_f32 v65, v89, v239
	v_dual_fmac_f32 v74, v39, v247 :: v_dual_fmac_f32 v75, v49, v247
	v_dual_fmac_f32 v84, v59, v247 :: v_dual_fmac_f32 v85, v69, v247
	v_dual_fmac_f32 v94, v79, v247 :: v_dual_fmac_f32 v95, v89, v247
	v_dual_fmac_f32 v44, v40, v240 :: v_dual_fmac_f32 v45, v50, v240
	v_dual_fmac_f32 v54, v60, v240 :: v_dual_fmac_f32 v55, v70, v240
	v_dual_fmac_f32 v64, v80, v240 :: v_dual_fmac_f32 v65, v90, v240
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v74, v40, v248 :: v_dual_fmac_f32 v75, v50, v248
	v_dual_fmac_f32 v84, v60, v248 :: v_dual_fmac_f32 v85, v70, v248
	v_dual_fmac_f32 v94, v80, v248 :: v_dual_fmac_f32 v95, v90, v248
	v_dual_fmac_f32 v44, v41, v241 :: v_dual_fmac_f32 v45, v51, v241
	v_dual_fmac_f32 v54, v61, v241 :: v_dual_fmac_f32 v55, v71, v241
	v_dual_fmac_f32 v64, v81, v241 :: v_dual_fmac_f32 v65, v91, v241
	v_dual_fmac_f32 v74, v41, v249 :: v_dual_fmac_f32 v75, v51, v249
	v_dual_fmac_f32 v84, v61, v249 :: v_dual_fmac_f32 v85, v71, v249
	v_dual_fmac_f32 v94, v81, v249 :: v_dual_fmac_f32 v95, v91, v249
	v_dual_fmac_f32 v44, v42, v242 :: v_dual_fmac_f32 v45, v52, v242
	v_dual_fmac_f32 v54, v62, v242 :: v_dual_fmac_f32 v55, v72, v242
	v_dual_fmac_f32 v64, v82, v242 :: v_dual_fmac_f32 v65, v92, v242
	v_dual_fmac_f32 v74, v42, v250 :: v_dual_fmac_f32 v75, v52, v250
	v_dual_fmac_f32 v84, v62, v250 :: v_dual_fmac_f32 v85, v72, v250
	v_dual_fmac_f32 v94, v82, v250 :: v_dual_fmac_f32 v95, v92, v250
	v_dual_fmac_f32 v44, v43, v243 :: v_dual_fmac_f32 v45, v53, v243
	v_dual_fmac_f32 v54, v63, v243 :: v_dual_fmac_f32 v55, v73, v243
	v_dual_fmac_f32 v64, v83, v243 :: v_dual_fmac_f32 v65, v93, v243
	v_dual_fmac_f32 v74, v43, v251 :: v_dual_fmac_f32 v75, v53, v251
	v_dual_fmac_f32 v84, v63, v251 :: v_dual_fmac_f32 v85, v73, v251
	v_dual_fmac_f32 v94, v83, v251 :: v_dual_fmac_f32 v95, v93, v251
	global_load_b128 v[236:239], v3, s[36:37]
	global_load_b128 v[240:243], v3, s[36:37] offset:16
	global_load_b128 v[244:247], v3, s[38:39]
	global_load_b128 v[248:251], v3, s[38:39] offset:16
	v_dual_add_f32 v174, v44, v174 :: v_dual_add_f32 v175, v175, v45
	v_dual_add_f32 v176, v54, v176 :: v_dual_add_f32 v177, v177, v55
	v_dual_add_f32 v178, v64, v178 :: v_dual_add_f32 v179, v179, v65
	v_dual_add_f32 v192, v74, v192 :: v_dual_add_f32 v193, v193, v75
	v_dual_add_f32 v194, v84, v194 :: v_dual_add_f32 v195, v195, v85
	v_dual_add_f32 v196, v94, v196 :: v_dual_add_f32 v197, v197, v95
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s2
	s_wait_loadcnt 0x0
	.Lrx_b6_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[204:207], v2, s[28:29]
	global_load_b128 v[208:211], v2, s[28:29] offset:16
	global_load_b128 v[212:215], v2, s[30:31]
	global_load_b128 v[216:219], v2, s[30:31] offset:16
	global_load_b128 v[220:223], v2, s[32:33]
	global_load_b128 v[224:227], v2, s[32:33] offset:16
	global_load_b128 v[228:231], v2, s[34:35]
	global_load_b128 v[232:235], v2, s[34:35] offset:16
	global_load_b128 v[236:239], v2, s[36:37]
	global_load_b128 v[240:243], v2, s[36:37] offset:16
	global_load_b128 v[244:247], v2, s[38:39]
	global_load_b128 v[248:251], v2, s[38:39] offset:16
	.Lrx_b6_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x16
	v_bfe_u32 v36, v28, 0, 4
	v_bfe_u32 v37, v28, 4, 4
	v_bfe_u32 v38, v28, 8, 4
	v_bfe_u32 v39, v28, 12, 4
	v_bfe_u32 v40, v28, 16, 4
	v_bfe_u32 v41, v28, 20, 4
	v_bfe_u32 v42, v28, 24, 4
	v_bfe_u32 v43, v28, 28, 4
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_cvt_f32_ubyte0_e32 v42, v42
	v_cvt_f32_ubyte0_e32 v43, v43
	v_fma_mix_f32 v36, v16, v36, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v16, v37, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v16, v38, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v16, v39, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v46, v29, 0, 4
	v_bfe_u32 v47, v29, 4, 4
	v_bfe_u32 v48, v29, 8, 4
	v_bfe_u32 v49, v29, 12, 4
	v_bfe_u32 v50, v29, 16, 4
	v_bfe_u32 v51, v29, 20, 4
	v_bfe_u32 v52, v29, 24, 4
	v_bfe_u32 v53, v29, 28, 4
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_cvt_f32_ubyte0_e32 v52, v52
	v_cvt_f32_ubyte0_e32 v53, v53
	v_fma_mix_f32 v46, v18, v46, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v18, v47, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v18, v48, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v18, v49, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v18, v50, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v18, v51, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v18, v52, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v18, v53, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v20, v21, v20, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v56, v30, 0, 4
	v_bfe_u32 v57, v30, 4, 4
	v_bfe_u32 v58, v30, 8, 4
	v_bfe_u32 v59, v30, 12, 4
	v_bfe_u32 v60, v30, 16, 4
	v_bfe_u32 v61, v30, 20, 4
	v_bfe_u32 v62, v30, 24, 4
	v_bfe_u32 v63, v30, 28, 4
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f32_ubyte0_e32 v62, v62
	v_cvt_f32_ubyte0_e32 v63, v63
	v_fma_mix_f32 v56, v20, v56, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v20, v57, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v20, v58, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v20, v59, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v20, v60, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v20, v61, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v20, v62, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v20, v63, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v22, v23, v22, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v66, v31, 0, 4
	v_bfe_u32 v67, v31, 4, 4
	v_bfe_u32 v68, v31, 8, 4
	v_bfe_u32 v69, v31, 12, 4
	v_bfe_u32 v70, v31, 16, 4
	v_bfe_u32 v71, v31, 20, 4
	v_bfe_u32 v72, v31, 24, 4
	v_bfe_u32 v73, v31, 28, 4
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_cvt_f32_ubyte0_e32 v72, v72
	v_cvt_f32_ubyte0_e32 v73, v73
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v22, v68, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v22, v69, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v22, v70, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v22, v71, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v22, v72, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v22, v73, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v24, v25, v24, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v76, v32, 0, 4
	v_bfe_u32 v77, v32, 4, 4
	v_bfe_u32 v78, v32, 8, 4
	v_bfe_u32 v79, v32, 12, 4
	v_bfe_u32 v80, v32, 16, 4
	v_bfe_u32 v81, v32, 20, 4
	v_bfe_u32 v82, v32, 24, 4
	v_bfe_u32 v83, v32, 28, 4
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v81, v81
	v_cvt_f32_ubyte0_e32 v82, v82
	v_cvt_f32_ubyte0_e32 v83, v83
	v_fma_mix_f32 v76, v24, v76, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v24, v77, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v24, v78, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v24, v79, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v80, v24, v80, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v81, v24, v81, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v82, v24, v82, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v83, v24, v83, v24 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v26, v27, v26, vcc_lo
	s_wait_loadcnt 0xc
	v_bfe_u32 v86, v33, 0, 4
	v_bfe_u32 v87, v33, 4, 4
	v_bfe_u32 v88, v33, 8, 4
	v_bfe_u32 v89, v33, 12, 4
	v_bfe_u32 v90, v33, 16, 4
	v_bfe_u32 v91, v33, 20, 4
	v_bfe_u32 v92, v33, 24, 4
	v_bfe_u32 v93, v33, 28, 4
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v91, v91
	v_cvt_f32_ubyte0_e32 v92, v92
	v_cvt_f32_ubyte0_e32 v93, v93
	v_fma_mix_f32 v86, v26, v86, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v87, v26, v87, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v88, v26, v88, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v89, v26, v89, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v90, v26, v90, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v91, v26, v91, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v92, v26, v92, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v93, v26, v93, v26 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v28, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v29, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[20:21], v12, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v30, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[22:23], v13, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v31, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[24:25], v14, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v32, v8, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[26:27], v15, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v33, v9, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v44, v36, v204 :: v_dual_mul_f32 v45, v47, v205
	v_dual_mul_f32 v54, v56, v204 :: v_dual_mul_f32 v55, v67, v205
	v_dual_mul_f32 v64, v76, v204 :: v_dual_mul_f32 v65, v87, v205
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v74, v36, v212 :: v_dual_mul_f32 v75, v47, v213
	v_dual_mul_f32 v84, v56, v212 :: v_dual_mul_f32 v85, v67, v213
	v_dual_mul_f32 v94, v76, v212 :: v_dual_mul_f32 v95, v87, v213
	v_dual_fmac_f32 v44, v37, v205 :: v_dual_fmac_f32 v45, v46, v204
	v_dual_fmac_f32 v54, v57, v205 :: v_dual_fmac_f32 v55, v66, v204
	v_dual_fmac_f32 v64, v77, v205 :: v_dual_fmac_f32 v65, v86, v204
	v_dual_fmac_f32 v74, v37, v213 :: v_dual_fmac_f32 v75, v46, v212
	v_dual_fmac_f32 v84, v57, v213 :: v_dual_fmac_f32 v85, v66, v212
	v_dual_fmac_f32 v94, v77, v213 :: v_dual_fmac_f32 v95, v86, v212
	v_dual_fmac_f32 v44, v38, v206 :: v_dual_fmac_f32 v45, v48, v206
	v_dual_fmac_f32 v54, v58, v206 :: v_dual_fmac_f32 v55, v68, v206
	v_dual_fmac_f32 v64, v78, v206 :: v_dual_fmac_f32 v65, v88, v206
	v_dual_fmac_f32 v74, v38, v214 :: v_dual_fmac_f32 v75, v48, v214
	v_dual_fmac_f32 v84, v58, v214 :: v_dual_fmac_f32 v85, v68, v214
	v_dual_fmac_f32 v94, v78, v214 :: v_dual_fmac_f32 v95, v88, v214
	v_dual_fmac_f32 v44, v39, v207 :: v_dual_fmac_f32 v45, v49, v207
	v_dual_fmac_f32 v54, v59, v207 :: v_dual_fmac_f32 v55, v69, v207
	v_dual_fmac_f32 v64, v79, v207 :: v_dual_fmac_f32 v65, v89, v207
	v_dual_fmac_f32 v74, v39, v215 :: v_dual_fmac_f32 v75, v49, v215
	v_dual_fmac_f32 v84, v59, v215 :: v_dual_fmac_f32 v85, v69, v215
	v_dual_fmac_f32 v94, v79, v215 :: v_dual_fmac_f32 v95, v89, v215
	v_dual_fmac_f32 v44, v40, v208 :: v_dual_fmac_f32 v45, v50, v208
	v_dual_fmac_f32 v54, v60, v208 :: v_dual_fmac_f32 v55, v70, v208
	v_dual_fmac_f32 v64, v80, v208 :: v_dual_fmac_f32 v65, v90, v208
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v74, v40, v216 :: v_dual_fmac_f32 v75, v50, v216
	v_dual_fmac_f32 v84, v60, v216 :: v_dual_fmac_f32 v85, v70, v216
	v_dual_fmac_f32 v94, v80, v216 :: v_dual_fmac_f32 v95, v90, v216
	v_dual_fmac_f32 v44, v41, v209 :: v_dual_fmac_f32 v45, v51, v209
	v_dual_fmac_f32 v54, v61, v209 :: v_dual_fmac_f32 v55, v71, v209
	v_dual_fmac_f32 v64, v81, v209 :: v_dual_fmac_f32 v65, v91, v209
	v_dual_fmac_f32 v74, v41, v217 :: v_dual_fmac_f32 v75, v51, v217
	v_dual_fmac_f32 v84, v61, v217 :: v_dual_fmac_f32 v85, v71, v217
	v_dual_fmac_f32 v94, v81, v217 :: v_dual_fmac_f32 v95, v91, v217
	v_dual_fmac_f32 v44, v42, v210 :: v_dual_fmac_f32 v45, v52, v210
	v_dual_fmac_f32 v54, v62, v210 :: v_dual_fmac_f32 v55, v72, v210
	v_dual_fmac_f32 v64, v82, v210 :: v_dual_fmac_f32 v65, v92, v210
	v_dual_fmac_f32 v74, v42, v218 :: v_dual_fmac_f32 v75, v52, v218
	v_dual_fmac_f32 v84, v62, v218 :: v_dual_fmac_f32 v85, v72, v218
	v_dual_fmac_f32 v94, v82, v218 :: v_dual_fmac_f32 v95, v92, v218
	v_dual_fmac_f32 v44, v43, v211 :: v_dual_fmac_f32 v45, v53, v211
	v_dual_fmac_f32 v54, v63, v211 :: v_dual_fmac_f32 v55, v73, v211
	v_dual_fmac_f32 v64, v83, v211 :: v_dual_fmac_f32 v65, v93, v211
	v_dual_fmac_f32 v74, v43, v219 :: v_dual_fmac_f32 v75, v53, v219
	v_dual_fmac_f32 v84, v63, v219 :: v_dual_fmac_f32 v85, v73, v219
	v_dual_fmac_f32 v94, v83, v219 :: v_dual_fmac_f32 v95, v93, v219
	global_load_b128 v[204:207], v3, s[28:29]
	global_load_b128 v[208:211], v3, s[28:29] offset:16
	global_load_b128 v[212:215], v3, s[30:31]
	global_load_b128 v[216:219], v3, s[30:31] offset:16
	v_dual_add_f32 v108, v44, v108 :: v_dual_add_f32 v109, v109, v45
	v_dual_add_f32 v110, v54, v110 :: v_dual_add_f32 v111, v111, v55
	v_dual_add_f32 v112, v64, v112 :: v_dual_add_f32 v113, v113, v65
	v_dual_add_f32 v126, v74, v126 :: v_dual_add_f32 v127, v127, v75
	v_dual_add_f32 v128, v84, v128 :: v_dual_add_f32 v129, v129, v85
	v_dual_add_f32 v130, v94, v130 :: v_dual_add_f32 v131, v131, v95
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v44, v36, v220 :: v_dual_mul_f32 v45, v47, v221
	v_dual_mul_f32 v54, v56, v220 :: v_dual_mul_f32 v55, v67, v221
	v_dual_mul_f32 v64, v76, v220 :: v_dual_mul_f32 v65, v87, v221
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v74, v36, v228 :: v_dual_mul_f32 v75, v47, v229
	v_dual_mul_f32 v84, v56, v228 :: v_dual_mul_f32 v85, v67, v229
	v_dual_mul_f32 v94, v76, v228 :: v_dual_mul_f32 v95, v87, v229
	v_dual_fmac_f32 v44, v37, v221 :: v_dual_fmac_f32 v45, v46, v220
	v_dual_fmac_f32 v54, v57, v221 :: v_dual_fmac_f32 v55, v66, v220
	v_dual_fmac_f32 v64, v77, v221 :: v_dual_fmac_f32 v65, v86, v220
	v_dual_fmac_f32 v74, v37, v229 :: v_dual_fmac_f32 v75, v46, v228
	v_dual_fmac_f32 v84, v57, v229 :: v_dual_fmac_f32 v85, v66, v228
	v_dual_fmac_f32 v94, v77, v229 :: v_dual_fmac_f32 v95, v86, v228
	v_dual_fmac_f32 v44, v38, v222 :: v_dual_fmac_f32 v45, v48, v222
	v_dual_fmac_f32 v54, v58, v222 :: v_dual_fmac_f32 v55, v68, v222
	v_dual_fmac_f32 v64, v78, v222 :: v_dual_fmac_f32 v65, v88, v222
	v_dual_fmac_f32 v74, v38, v230 :: v_dual_fmac_f32 v75, v48, v230
	v_dual_fmac_f32 v84, v58, v230 :: v_dual_fmac_f32 v85, v68, v230
	v_dual_fmac_f32 v94, v78, v230 :: v_dual_fmac_f32 v95, v88, v230
	v_dual_fmac_f32 v44, v39, v223 :: v_dual_fmac_f32 v45, v49, v223
	v_dual_fmac_f32 v54, v59, v223 :: v_dual_fmac_f32 v55, v69, v223
	v_dual_fmac_f32 v64, v79, v223 :: v_dual_fmac_f32 v65, v89, v223
	v_dual_fmac_f32 v74, v39, v231 :: v_dual_fmac_f32 v75, v49, v231
	v_dual_fmac_f32 v84, v59, v231 :: v_dual_fmac_f32 v85, v69, v231
	v_dual_fmac_f32 v94, v79, v231 :: v_dual_fmac_f32 v95, v89, v231
	v_dual_fmac_f32 v44, v40, v224 :: v_dual_fmac_f32 v45, v50, v224
	v_dual_fmac_f32 v54, v60, v224 :: v_dual_fmac_f32 v55, v70, v224
	v_dual_fmac_f32 v64, v80, v224 :: v_dual_fmac_f32 v65, v90, v224
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v74, v40, v232 :: v_dual_fmac_f32 v75, v50, v232
	v_dual_fmac_f32 v84, v60, v232 :: v_dual_fmac_f32 v85, v70, v232
	v_dual_fmac_f32 v94, v80, v232 :: v_dual_fmac_f32 v95, v90, v232
	v_dual_fmac_f32 v44, v41, v225 :: v_dual_fmac_f32 v45, v51, v225
	v_dual_fmac_f32 v54, v61, v225 :: v_dual_fmac_f32 v55, v71, v225
	v_dual_fmac_f32 v64, v81, v225 :: v_dual_fmac_f32 v65, v91, v225
	v_dual_fmac_f32 v74, v41, v233 :: v_dual_fmac_f32 v75, v51, v233
	v_dual_fmac_f32 v84, v61, v233 :: v_dual_fmac_f32 v85, v71, v233
	v_dual_fmac_f32 v94, v81, v233 :: v_dual_fmac_f32 v95, v91, v233
	v_dual_fmac_f32 v44, v42, v226 :: v_dual_fmac_f32 v45, v52, v226
	v_dual_fmac_f32 v54, v62, v226 :: v_dual_fmac_f32 v55, v72, v226
	v_dual_fmac_f32 v64, v82, v226 :: v_dual_fmac_f32 v65, v92, v226
	v_dual_fmac_f32 v74, v42, v234 :: v_dual_fmac_f32 v75, v52, v234
	v_dual_fmac_f32 v84, v62, v234 :: v_dual_fmac_f32 v85, v72, v234
	v_dual_fmac_f32 v94, v82, v234 :: v_dual_fmac_f32 v95, v92, v234
	v_dual_fmac_f32 v44, v43, v227 :: v_dual_fmac_f32 v45, v53, v227
	v_dual_fmac_f32 v54, v63, v227 :: v_dual_fmac_f32 v55, v73, v227
	v_dual_fmac_f32 v64, v83, v227 :: v_dual_fmac_f32 v65, v93, v227
	v_dual_fmac_f32 v74, v43, v235 :: v_dual_fmac_f32 v75, v53, v235
	v_dual_fmac_f32 v84, v63, v235 :: v_dual_fmac_f32 v85, v73, v235
	v_dual_fmac_f32 v94, v83, v235 :: v_dual_fmac_f32 v95, v93, v235
	global_load_b128 v[220:223], v3, s[32:33]
	global_load_b128 v[224:227], v3, s[32:33] offset:16
	global_load_b128 v[228:231], v3, s[34:35]
	global_load_b128 v[232:235], v3, s[34:35] offset:16
	v_dual_add_f32 v144, v44, v144 :: v_dual_add_f32 v145, v145, v45
	v_dual_add_f32 v146, v54, v146 :: v_dual_add_f32 v147, v147, v55
	v_dual_add_f32 v148, v64, v148 :: v_dual_add_f32 v149, v149, v65
	v_dual_add_f32 v162, v74, v162 :: v_dual_add_f32 v163, v163, v75
	v_dual_add_f32 v164, v84, v164 :: v_dual_add_f32 v165, v165, v85
	v_dual_add_f32 v166, v94, v166 :: v_dual_add_f32 v167, v167, v95
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v44, v36, v236 :: v_dual_mul_f32 v45, v47, v237
	v_dual_mul_f32 v54, v56, v236 :: v_dual_mul_f32 v55, v67, v237
	v_dual_mul_f32 v64, v76, v236 :: v_dual_mul_f32 v65, v87, v237
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v74, v36, v244 :: v_dual_mul_f32 v75, v47, v245
	v_dual_mul_f32 v84, v56, v244 :: v_dual_mul_f32 v85, v67, v245
	v_dual_mul_f32 v94, v76, v244 :: v_dual_mul_f32 v95, v87, v245
	v_dual_fmac_f32 v44, v37, v237 :: v_dual_fmac_f32 v45, v46, v236
	v_dual_fmac_f32 v54, v57, v237 :: v_dual_fmac_f32 v55, v66, v236
	v_dual_fmac_f32 v64, v77, v237 :: v_dual_fmac_f32 v65, v86, v236
	v_dual_fmac_f32 v74, v37, v245 :: v_dual_fmac_f32 v75, v46, v244
	v_dual_fmac_f32 v84, v57, v245 :: v_dual_fmac_f32 v85, v66, v244
	v_dual_fmac_f32 v94, v77, v245 :: v_dual_fmac_f32 v95, v86, v244
	v_dual_fmac_f32 v44, v38, v238 :: v_dual_fmac_f32 v45, v48, v238
	v_dual_fmac_f32 v54, v58, v238 :: v_dual_fmac_f32 v55, v68, v238
	v_dual_fmac_f32 v64, v78, v238 :: v_dual_fmac_f32 v65, v88, v238
	v_dual_fmac_f32 v74, v38, v246 :: v_dual_fmac_f32 v75, v48, v246
	v_dual_fmac_f32 v84, v58, v246 :: v_dual_fmac_f32 v85, v68, v246
	v_dual_fmac_f32 v94, v78, v246 :: v_dual_fmac_f32 v95, v88, v246
	v_dual_fmac_f32 v44, v39, v239 :: v_dual_fmac_f32 v45, v49, v239
	v_dual_fmac_f32 v54, v59, v239 :: v_dual_fmac_f32 v55, v69, v239
	v_dual_fmac_f32 v64, v79, v239 :: v_dual_fmac_f32 v65, v89, v239
	v_dual_fmac_f32 v74, v39, v247 :: v_dual_fmac_f32 v75, v49, v247
	v_dual_fmac_f32 v84, v59, v247 :: v_dual_fmac_f32 v85, v69, v247
	v_dual_fmac_f32 v94, v79, v247 :: v_dual_fmac_f32 v95, v89, v247
	v_dual_fmac_f32 v44, v40, v240 :: v_dual_fmac_f32 v45, v50, v240
	v_dual_fmac_f32 v54, v60, v240 :: v_dual_fmac_f32 v55, v70, v240
	v_dual_fmac_f32 v64, v80, v240 :: v_dual_fmac_f32 v65, v90, v240
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v74, v40, v248 :: v_dual_fmac_f32 v75, v50, v248
	v_dual_fmac_f32 v84, v60, v248 :: v_dual_fmac_f32 v85, v70, v248
	v_dual_fmac_f32 v94, v80, v248 :: v_dual_fmac_f32 v95, v90, v248
	v_dual_fmac_f32 v44, v41, v241 :: v_dual_fmac_f32 v45, v51, v241
	v_dual_fmac_f32 v54, v61, v241 :: v_dual_fmac_f32 v55, v71, v241
	v_dual_fmac_f32 v64, v81, v241 :: v_dual_fmac_f32 v65, v91, v241
	v_dual_fmac_f32 v74, v41, v249 :: v_dual_fmac_f32 v75, v51, v249
	v_dual_fmac_f32 v84, v61, v249 :: v_dual_fmac_f32 v85, v71, v249
	v_dual_fmac_f32 v94, v81, v249 :: v_dual_fmac_f32 v95, v91, v249
	v_dual_fmac_f32 v44, v42, v242 :: v_dual_fmac_f32 v45, v52, v242
	v_dual_fmac_f32 v54, v62, v242 :: v_dual_fmac_f32 v55, v72, v242
	v_dual_fmac_f32 v64, v82, v242 :: v_dual_fmac_f32 v65, v92, v242
	v_dual_fmac_f32 v74, v42, v250 :: v_dual_fmac_f32 v75, v52, v250
	v_dual_fmac_f32 v84, v62, v250 :: v_dual_fmac_f32 v85, v72, v250
	v_dual_fmac_f32 v94, v82, v250 :: v_dual_fmac_f32 v95, v92, v250
	v_dual_fmac_f32 v44, v43, v243 :: v_dual_fmac_f32 v45, v53, v243
	v_dual_fmac_f32 v54, v63, v243 :: v_dual_fmac_f32 v55, v73, v243
	v_dual_fmac_f32 v64, v83, v243 :: v_dual_fmac_f32 v65, v93, v243
	v_dual_fmac_f32 v74, v43, v251 :: v_dual_fmac_f32 v75, v53, v251
	v_dual_fmac_f32 v84, v63, v251 :: v_dual_fmac_f32 v85, v73, v251
	v_dual_fmac_f32 v94, v83, v251 :: v_dual_fmac_f32 v95, v93, v251
	global_load_b128 v[236:239], v3, s[36:37]
	global_load_b128 v[240:243], v3, s[36:37] offset:16
	global_load_b128 v[244:247], v3, s[38:39]
	global_load_b128 v[248:251], v3, s[38:39] offset:16
	v_dual_add_f32 v180, v44, v180 :: v_dual_add_f32 v181, v181, v45
	v_dual_add_f32 v182, v54, v182 :: v_dual_add_f32 v183, v183, v55
	v_dual_add_f32 v184, v64, v184 :: v_dual_add_f32 v185, v185, v65
	v_dual_add_f32 v198, v74, v198 :: v_dual_add_f32 v199, v199, v75
	v_dual_add_f32 v200, v84, v200 :: v_dual_add_f32 v201, v201, v85
	v_dual_add_f32 v202, v94, v202 :: v_dual_add_f32 v203, v203, v95
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s3
	s_wait_loadcnt 0x0
	.Lrx_b6_s3_done:
	v_dual_add_f32 v102, v102, v108 :: v_dual_add_f32 v103, v103, v109
	v_dual_add_f32 v104, v104, v110 :: v_dual_add_f32 v105, v105, v111
	v_dual_add_f32 v106, v106, v112 :: v_dual_add_f32 v107, v107, v113
	v_dual_add_f32 v120, v120, v126 :: v_dual_add_f32 v121, v121, v127
	v_dual_add_f32 v122, v122, v128 :: v_dual_add_f32 v123, v123, v129
	v_dual_add_f32 v124, v124, v130 :: v_dual_add_f32 v125, v125, v131
	v_dual_add_f32 v138, v138, v144 :: v_dual_add_f32 v139, v139, v145
	v_dual_add_f32 v140, v140, v146 :: v_dual_add_f32 v141, v141, v147
	v_dual_add_f32 v142, v142, v148 :: v_dual_add_f32 v143, v143, v149
	v_dual_add_f32 v156, v156, v162 :: v_dual_add_f32 v157, v157, v163
	v_dual_add_f32 v158, v158, v164 :: v_dual_add_f32 v159, v159, v165
	v_dual_add_f32 v160, v160, v166 :: v_dual_add_f32 v161, v161, v167
	v_dual_add_f32 v174, v174, v180 :: v_dual_add_f32 v175, v175, v181
	v_dual_add_f32 v176, v176, v182 :: v_dual_add_f32 v177, v177, v183
	v_dual_add_f32 v178, v178, v184 :: v_dual_add_f32 v179, v179, v185
	v_dual_add_f32 v192, v192, v198 :: v_dual_add_f32 v193, v193, v199
	v_dual_add_f32 v194, v194, v200 :: v_dual_add_f32 v195, v195, v201
	v_dual_add_f32 v196, v196, v202 :: v_dual_add_f32 v197, v197, v203
	v_dual_add_f32 v96, v96, v102 :: v_dual_add_f32 v97, v97, v103
	v_dual_add_f32 v98, v98, v104 :: v_dual_add_f32 v99, v99, v105
	v_dual_add_f32 v100, v100, v106 :: v_dual_add_f32 v101, v101, v107
	v_dual_add_f32 v114, v114, v120 :: v_dual_add_f32 v115, v115, v121
	v_dual_add_f32 v116, v116, v122 :: v_dual_add_f32 v117, v117, v123
	v_dual_add_f32 v118, v118, v124 :: v_dual_add_f32 v119, v119, v125
	v_dual_add_f32 v132, v132, v138 :: v_dual_add_f32 v133, v133, v139
	v_dual_add_f32 v134, v134, v140 :: v_dual_add_f32 v135, v135, v141
	v_dual_add_f32 v136, v136, v142 :: v_dual_add_f32 v137, v137, v143
	v_dual_add_f32 v150, v150, v156 :: v_dual_add_f32 v151, v151, v157
	v_dual_add_f32 v152, v152, v158 :: v_dual_add_f32 v153, v153, v159
	v_dual_add_f32 v154, v154, v160 :: v_dual_add_f32 v155, v155, v161
	v_dual_add_f32 v168, v168, v174 :: v_dual_add_f32 v169, v169, v175
	v_dual_add_f32 v170, v170, v176 :: v_dual_add_f32 v171, v171, v177
	v_dual_add_f32 v172, v172, v178 :: v_dual_add_f32 v173, v173, v179
	v_dual_add_f32 v186, v186, v192 :: v_dual_add_f32 v187, v187, v193
	v_dual_add_f32 v188, v188, v194 :: v_dual_add_f32 v189, v189, v195
	v_dual_add_f32 v190, v190, v196 :: v_dual_add_f32 v191, v191, v197
	ds_swizzle_b32 v204, v96 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v205, v97 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v206, v98 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v207, v99 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v208, v100 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v209, v101 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v210, v114 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v211, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v212, v116 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v213, v117 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v214, v118 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v215, v119 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v216, v132 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v217, v133 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v218, v134 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v219, v135 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v220, v136 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v221, v137 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v222, v150 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v223, v151 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v224, v152 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v225, v153 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v226, v154 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v227, v155 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v228, v168 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v229, v169 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v230, v170 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v231, v171 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v232, v172 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v233, v173 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v234, v186 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v235, v187 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v236, v188 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v237, v189 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v238, v190 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v239, v191 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x23
	v_add_f32_e32 v96, v96, v204
	s_wait_dscnt 0x22
	v_add_f32_e32 v97, v97, v205
	s_wait_dscnt 0x21
	v_add_f32_e32 v98, v98, v206
	s_wait_dscnt 0x20
	v_add_f32_e32 v99, v99, v207
	s_wait_dscnt 0x1f
	v_add_f32_e32 v100, v100, v208
	s_wait_dscnt 0x1e
	v_add_f32_e32 v101, v101, v209
	s_wait_dscnt 0x1d
	v_add_f32_e32 v114, v114, v210
	s_wait_dscnt 0x1c
	v_add_f32_e32 v115, v115, v211
	s_wait_dscnt 0x1b
	v_add_f32_e32 v116, v116, v212
	s_wait_dscnt 0x1a
	v_add_f32_e32 v117, v117, v213
	s_wait_dscnt 0x19
	v_add_f32_e32 v118, v118, v214
	s_wait_dscnt 0x18
	v_add_f32_e32 v119, v119, v215
	s_wait_dscnt 0x17
	v_add_f32_e32 v132, v132, v216
	s_wait_dscnt 0x16
	v_add_f32_e32 v133, v133, v217
	s_wait_dscnt 0x15
	v_add_f32_e32 v134, v134, v218
	s_wait_dscnt 0x14
	v_add_f32_e32 v135, v135, v219
	s_wait_dscnt 0x13
	v_add_f32_e32 v136, v136, v220
	s_wait_dscnt 0x12
	v_add_f32_e32 v137, v137, v221
	s_wait_dscnt 0x11
	v_add_f32_e32 v150, v150, v222
	s_wait_dscnt 0x10
	v_add_f32_e32 v151, v151, v223
	s_wait_dscnt 0xf
	v_add_f32_e32 v152, v152, v224
	s_wait_dscnt 0xe
	v_add_f32_e32 v153, v153, v225
	s_wait_dscnt 0xd
	v_add_f32_e32 v154, v154, v226
	s_wait_dscnt 0xc
	v_add_f32_e32 v155, v155, v227
	s_wait_dscnt 0xb
	v_add_f32_e32 v168, v168, v228
	s_wait_dscnt 0xa
	v_add_f32_e32 v169, v169, v229
	s_wait_dscnt 0x9
	v_add_f32_e32 v170, v170, v230
	s_wait_dscnt 0x8
	v_add_f32_e32 v171, v171, v231
	s_wait_dscnt 0x7
	v_add_f32_e32 v172, v172, v232
	s_wait_dscnt 0x6
	v_add_f32_e32 v173, v173, v233
	s_wait_dscnt 0x5
	v_add_f32_e32 v186, v186, v234
	s_wait_dscnt 0x4
	v_add_f32_e32 v187, v187, v235
	s_wait_dscnt 0x3
	v_add_f32_e32 v188, v188, v236
	s_wait_dscnt 0x2
	v_add_f32_e32 v189, v189, v237
	s_wait_dscnt 0x1
	v_add_f32_e32 v190, v190, v238
	s_wait_dscnt 0x0
	v_add_f32_e32 v191, v191, v239
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v204, v3, v96
	ds_bpermute_b32 v205, v3, v97
	ds_bpermute_b32 v206, v3, v98
	ds_bpermute_b32 v207, v3, v99
	ds_bpermute_b32 v208, v3, v100
	ds_bpermute_b32 v209, v3, v101
	ds_bpermute_b32 v210, v3, v114
	ds_bpermute_b32 v211, v3, v115
	ds_bpermute_b32 v212, v3, v116
	ds_bpermute_b32 v213, v3, v117
	ds_bpermute_b32 v214, v3, v118
	ds_bpermute_b32 v215, v3, v119
	ds_bpermute_b32 v216, v3, v132
	ds_bpermute_b32 v217, v3, v133
	ds_bpermute_b32 v218, v3, v134
	ds_bpermute_b32 v219, v3, v135
	ds_bpermute_b32 v220, v3, v136
	ds_bpermute_b32 v221, v3, v137
	ds_bpermute_b32 v222, v3, v150
	ds_bpermute_b32 v223, v3, v151
	ds_bpermute_b32 v224, v3, v152
	ds_bpermute_b32 v225, v3, v153
	ds_bpermute_b32 v226, v3, v154
	ds_bpermute_b32 v227, v3, v155
	ds_bpermute_b32 v228, v3, v168
	ds_bpermute_b32 v229, v3, v169
	ds_bpermute_b32 v230, v3, v170
	ds_bpermute_b32 v231, v3, v171
	ds_bpermute_b32 v232, v3, v172
	ds_bpermute_b32 v233, v3, v173
	ds_bpermute_b32 v234, v3, v186
	ds_bpermute_b32 v235, v3, v187
	ds_bpermute_b32 v236, v3, v188
	ds_bpermute_b32 v237, v3, v189
	ds_bpermute_b32 v238, v3, v190
	ds_bpermute_b32 v239, v3, v191
	s_wait_dscnt 0x23
	v_add_f32_e32 v96, v96, v204
	s_wait_dscnt 0x22
	v_add_f32_e32 v97, v97, v205
	s_wait_dscnt 0x21
	v_add_f32_e32 v98, v98, v206
	s_wait_dscnt 0x20
	v_add_f32_e32 v99, v99, v207
	s_wait_dscnt 0x1f
	v_add_f32_e32 v100, v100, v208
	s_wait_dscnt 0x1e
	v_add_f32_e32 v101, v101, v209
	s_wait_dscnt 0x1d
	v_add_f32_e32 v114, v114, v210
	s_wait_dscnt 0x1c
	v_add_f32_e32 v115, v115, v211
	s_wait_dscnt 0x1b
	v_add_f32_e32 v116, v116, v212
	s_wait_dscnt 0x1a
	v_add_f32_e32 v117, v117, v213
	s_wait_dscnt 0x19
	v_add_f32_e32 v118, v118, v214
	s_wait_dscnt 0x18
	v_add_f32_e32 v119, v119, v215
	s_wait_dscnt 0x17
	v_add_f32_e32 v132, v132, v216
	s_wait_dscnt 0x16
	v_add_f32_e32 v133, v133, v217
	s_wait_dscnt 0x15
	v_add_f32_e32 v134, v134, v218
	s_wait_dscnt 0x14
	v_add_f32_e32 v135, v135, v219
	s_wait_dscnt 0x13
	v_add_f32_e32 v136, v136, v220
	s_wait_dscnt 0x12
	v_add_f32_e32 v137, v137, v221
	s_wait_dscnt 0x11
	v_add_f32_e32 v150, v150, v222
	s_wait_dscnt 0x10
	v_add_f32_e32 v151, v151, v223
	s_wait_dscnt 0xf
	v_add_f32_e32 v152, v152, v224
	s_wait_dscnt 0xe
	v_add_f32_e32 v153, v153, v225
	s_wait_dscnt 0xd
	v_add_f32_e32 v154, v154, v226
	s_wait_dscnt 0xc
	v_add_f32_e32 v155, v155, v227
	s_wait_dscnt 0xb
	v_add_f32_e32 v168, v168, v228
	s_wait_dscnt 0xa
	v_add_f32_e32 v169, v169, v229
	s_wait_dscnt 0x9
	v_add_f32_e32 v170, v170, v230
	s_wait_dscnt 0x8
	v_add_f32_e32 v171, v171, v231
	s_wait_dscnt 0x7
	v_add_f32_e32 v172, v172, v232
	s_wait_dscnt 0x6
	v_add_f32_e32 v173, v173, v233
	s_wait_dscnt 0x5
	v_add_f32_e32 v186, v186, v234
	s_wait_dscnt 0x4
	v_add_f32_e32 v187, v187, v235
	s_wait_dscnt 0x3
	v_add_f32_e32 v188, v188, v236
	s_wait_dscnt 0x2
	v_add_f32_e32 v189, v189, v237
	s_wait_dscnt 0x1
	v_add_f32_e32 v190, v190, v238
	s_wait_dscnt 0x0
	v_add_f32_e32 v191, v191, v239
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v204, v3, v96
	ds_bpermute_b32 v205, v3, v97
	ds_bpermute_b32 v206, v3, v98
	ds_bpermute_b32 v207, v3, v99
	ds_bpermute_b32 v208, v3, v100
	ds_bpermute_b32 v209, v3, v101
	ds_bpermute_b32 v210, v3, v114
	ds_bpermute_b32 v211, v3, v115
	ds_bpermute_b32 v212, v3, v116
	ds_bpermute_b32 v213, v3, v117
	ds_bpermute_b32 v214, v3, v118
	ds_bpermute_b32 v215, v3, v119
	ds_bpermute_b32 v216, v3, v132
	ds_bpermute_b32 v217, v3, v133
	ds_bpermute_b32 v218, v3, v134
	ds_bpermute_b32 v219, v3, v135
	ds_bpermute_b32 v220, v3, v136
	ds_bpermute_b32 v221, v3, v137
	ds_bpermute_b32 v222, v3, v150
	ds_bpermute_b32 v223, v3, v151
	ds_bpermute_b32 v224, v3, v152
	ds_bpermute_b32 v225, v3, v153
	ds_bpermute_b32 v226, v3, v154
	ds_bpermute_b32 v227, v3, v155
	ds_bpermute_b32 v228, v3, v168
	ds_bpermute_b32 v229, v3, v169
	ds_bpermute_b32 v230, v3, v170
	ds_bpermute_b32 v231, v3, v171
	ds_bpermute_b32 v232, v3, v172
	ds_bpermute_b32 v233, v3, v173
	ds_bpermute_b32 v234, v3, v186
	ds_bpermute_b32 v235, v3, v187
	ds_bpermute_b32 v236, v3, v188
	ds_bpermute_b32 v237, v3, v189
	ds_bpermute_b32 v238, v3, v190
	ds_bpermute_b32 v239, v3, v191
	s_wait_dscnt 0x23
	v_add_f32_e32 v96, v96, v204
	s_wait_dscnt 0x22
	v_add_f32_e32 v97, v97, v205
	s_wait_dscnt 0x21
	v_add_f32_e32 v98, v98, v206
	s_wait_dscnt 0x20
	v_add_f32_e32 v99, v99, v207
	s_wait_dscnt 0x1f
	v_add_f32_e32 v100, v100, v208
	s_wait_dscnt 0x1e
	v_add_f32_e32 v101, v101, v209
	s_wait_dscnt 0x1d
	v_add_f32_e32 v114, v114, v210
	s_wait_dscnt 0x1c
	v_add_f32_e32 v115, v115, v211
	s_wait_dscnt 0x1b
	v_add_f32_e32 v116, v116, v212
	s_wait_dscnt 0x1a
	v_add_f32_e32 v117, v117, v213
	s_wait_dscnt 0x19
	v_add_f32_e32 v118, v118, v214
	s_wait_dscnt 0x18
	v_add_f32_e32 v119, v119, v215
	s_wait_dscnt 0x17
	v_add_f32_e32 v132, v132, v216
	s_wait_dscnt 0x16
	v_add_f32_e32 v133, v133, v217
	s_wait_dscnt 0x15
	v_add_f32_e32 v134, v134, v218
	s_wait_dscnt 0x14
	v_add_f32_e32 v135, v135, v219
	s_wait_dscnt 0x13
	v_add_f32_e32 v136, v136, v220
	s_wait_dscnt 0x12
	v_add_f32_e32 v137, v137, v221
	s_wait_dscnt 0x11
	v_add_f32_e32 v150, v150, v222
	s_wait_dscnt 0x10
	v_add_f32_e32 v151, v151, v223
	s_wait_dscnt 0xf
	v_add_f32_e32 v152, v152, v224
	s_wait_dscnt 0xe
	v_add_f32_e32 v153, v153, v225
	s_wait_dscnt 0xd
	v_add_f32_e32 v154, v154, v226
	s_wait_dscnt 0xc
	v_add_f32_e32 v155, v155, v227
	s_wait_dscnt 0xb
	v_add_f32_e32 v168, v168, v228
	s_wait_dscnt 0xa
	v_add_f32_e32 v169, v169, v229
	s_wait_dscnt 0x9
	v_add_f32_e32 v170, v170, v230
	s_wait_dscnt 0x8
	v_add_f32_e32 v171, v171, v231
	s_wait_dscnt 0x7
	v_add_f32_e32 v172, v172, v232
	s_wait_dscnt 0x6
	v_add_f32_e32 v173, v173, v233
	s_wait_dscnt 0x5
	v_add_f32_e32 v186, v186, v234
	s_wait_dscnt 0x4
	v_add_f32_e32 v187, v187, v235
	s_wait_dscnt 0x3
	v_add_f32_e32 v188, v188, v236
	s_wait_dscnt 0x2
	v_add_f32_e32 v189, v189, v237
	s_wait_dscnt 0x1
	v_add_f32_e32 v190, v190, v238
	s_wait_dscnt 0x0
	v_add_f32_e32 v191, v191, v239
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v204, v3, v96
	ds_bpermute_b32 v205, v3, v97
	ds_bpermute_b32 v206, v3, v98
	ds_bpermute_b32 v207, v3, v99
	ds_bpermute_b32 v208, v3, v100
	ds_bpermute_b32 v209, v3, v101
	ds_bpermute_b32 v210, v3, v114
	ds_bpermute_b32 v211, v3, v115
	ds_bpermute_b32 v212, v3, v116
	ds_bpermute_b32 v213, v3, v117
	ds_bpermute_b32 v214, v3, v118
	ds_bpermute_b32 v215, v3, v119
	ds_bpermute_b32 v216, v3, v132
	ds_bpermute_b32 v217, v3, v133
	ds_bpermute_b32 v218, v3, v134
	ds_bpermute_b32 v219, v3, v135
	ds_bpermute_b32 v220, v3, v136
	ds_bpermute_b32 v221, v3, v137
	ds_bpermute_b32 v222, v3, v150
	ds_bpermute_b32 v223, v3, v151
	ds_bpermute_b32 v224, v3, v152
	ds_bpermute_b32 v225, v3, v153
	ds_bpermute_b32 v226, v3, v154
	ds_bpermute_b32 v227, v3, v155
	ds_bpermute_b32 v228, v3, v168
	ds_bpermute_b32 v229, v3, v169
	ds_bpermute_b32 v230, v3, v170
	ds_bpermute_b32 v231, v3, v171
	ds_bpermute_b32 v232, v3, v172
	ds_bpermute_b32 v233, v3, v173
	ds_bpermute_b32 v234, v3, v186
	ds_bpermute_b32 v235, v3, v187
	ds_bpermute_b32 v236, v3, v188
	ds_bpermute_b32 v237, v3, v189
	ds_bpermute_b32 v238, v3, v190
	ds_bpermute_b32 v239, v3, v191
	s_wait_dscnt 0x23
	v_add_f32_e32 v96, v96, v204
	s_wait_dscnt 0x22
	v_add_f32_e32 v97, v97, v205
	s_wait_dscnt 0x21
	v_add_f32_e32 v98, v98, v206
	s_wait_dscnt 0x20
	v_add_f32_e32 v99, v99, v207
	s_wait_dscnt 0x1f
	v_add_f32_e32 v100, v100, v208
	s_wait_dscnt 0x1e
	v_add_f32_e32 v101, v101, v209
	s_wait_dscnt 0x1d
	v_add_f32_e32 v114, v114, v210
	s_wait_dscnt 0x1c
	v_add_f32_e32 v115, v115, v211
	s_wait_dscnt 0x1b
	v_add_f32_e32 v116, v116, v212
	s_wait_dscnt 0x1a
	v_add_f32_e32 v117, v117, v213
	s_wait_dscnt 0x19
	v_add_f32_e32 v118, v118, v214
	s_wait_dscnt 0x18
	v_add_f32_e32 v119, v119, v215
	s_wait_dscnt 0x17
	v_add_f32_e32 v132, v132, v216
	s_wait_dscnt 0x16
	v_add_f32_e32 v133, v133, v217
	s_wait_dscnt 0x15
	v_add_f32_e32 v134, v134, v218
	s_wait_dscnt 0x14
	v_add_f32_e32 v135, v135, v219
	s_wait_dscnt 0x13
	v_add_f32_e32 v136, v136, v220
	s_wait_dscnt 0x12
	v_add_f32_e32 v137, v137, v221
	s_wait_dscnt 0x11
	v_add_f32_e32 v150, v150, v222
	s_wait_dscnt 0x10
	v_add_f32_e32 v151, v151, v223
	s_wait_dscnt 0xf
	v_add_f32_e32 v152, v152, v224
	s_wait_dscnt 0xe
	v_add_f32_e32 v153, v153, v225
	s_wait_dscnt 0xd
	v_add_f32_e32 v154, v154, v226
	s_wait_dscnt 0xc
	v_add_f32_e32 v155, v155, v227
	s_wait_dscnt 0xb
	v_add_f32_e32 v168, v168, v228
	s_wait_dscnt 0xa
	v_add_f32_e32 v169, v169, v229
	s_wait_dscnt 0x9
	v_add_f32_e32 v170, v170, v230
	s_wait_dscnt 0x8
	v_add_f32_e32 v171, v171, v231
	s_wait_dscnt 0x7
	v_add_f32_e32 v172, v172, v232
	s_wait_dscnt 0x6
	v_add_f32_e32 v173, v173, v233
	s_wait_dscnt 0x5
	v_add_f32_e32 v186, v186, v234
	s_wait_dscnt 0x4
	v_add_f32_e32 v187, v187, v235
	s_wait_dscnt 0x3
	v_add_f32_e32 v188, v188, v236
	s_wait_dscnt 0x2
	v_add_f32_e32 v189, v189, v237
	s_wait_dscnt 0x1
	v_add_f32_e32 v190, v190, v238
	s_wait_dscnt 0x0
	v_add_f32_e32 v191, v191, v239
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v204, v3, v96
	ds_bpermute_b32 v205, v3, v97
	ds_bpermute_b32 v206, v3, v98
	ds_bpermute_b32 v207, v3, v99
	ds_bpermute_b32 v208, v3, v100
	ds_bpermute_b32 v209, v3, v101
	ds_bpermute_b32 v210, v3, v114
	ds_bpermute_b32 v211, v3, v115
	ds_bpermute_b32 v212, v3, v116
	ds_bpermute_b32 v213, v3, v117
	ds_bpermute_b32 v214, v3, v118
	ds_bpermute_b32 v215, v3, v119
	ds_bpermute_b32 v216, v3, v132
	ds_bpermute_b32 v217, v3, v133
	ds_bpermute_b32 v218, v3, v134
	ds_bpermute_b32 v219, v3, v135
	ds_bpermute_b32 v220, v3, v136
	ds_bpermute_b32 v221, v3, v137
	ds_bpermute_b32 v222, v3, v150
	ds_bpermute_b32 v223, v3, v151
	ds_bpermute_b32 v224, v3, v152
	ds_bpermute_b32 v225, v3, v153
	ds_bpermute_b32 v226, v3, v154
	ds_bpermute_b32 v227, v3, v155
	ds_bpermute_b32 v228, v3, v168
	ds_bpermute_b32 v229, v3, v169
	ds_bpermute_b32 v230, v3, v170
	ds_bpermute_b32 v231, v3, v171
	ds_bpermute_b32 v232, v3, v172
	ds_bpermute_b32 v233, v3, v173
	ds_bpermute_b32 v234, v3, v186
	ds_bpermute_b32 v235, v3, v187
	ds_bpermute_b32 v236, v3, v188
	ds_bpermute_b32 v237, v3, v189
	ds_bpermute_b32 v238, v3, v190
	ds_bpermute_b32 v239, v3, v191
	s_wait_dscnt 0x23
	v_add_f32_e32 v96, v96, v204
	s_wait_dscnt 0x22
	v_add_f32_e32 v97, v97, v205
	s_wait_dscnt 0x21
	v_add_f32_e32 v98, v98, v206
	s_wait_dscnt 0x20
	v_add_f32_e32 v99, v99, v207
	s_wait_dscnt 0x1f
	v_add_f32_e32 v100, v100, v208
	s_wait_dscnt 0x1e
	v_add_f32_e32 v101, v101, v209
	s_wait_dscnt 0x1d
	v_add_f32_e32 v114, v114, v210
	s_wait_dscnt 0x1c
	v_add_f32_e32 v115, v115, v211
	s_wait_dscnt 0x1b
	v_add_f32_e32 v116, v116, v212
	s_wait_dscnt 0x1a
	v_add_f32_e32 v117, v117, v213
	s_wait_dscnt 0x19
	v_add_f32_e32 v118, v118, v214
	s_wait_dscnt 0x18
	v_add_f32_e32 v119, v119, v215
	s_wait_dscnt 0x17
	v_add_f32_e32 v132, v132, v216
	s_wait_dscnt 0x16
	v_add_f32_e32 v133, v133, v217
	s_wait_dscnt 0x15
	v_add_f32_e32 v134, v134, v218
	s_wait_dscnt 0x14
	v_add_f32_e32 v135, v135, v219
	s_wait_dscnt 0x13
	v_add_f32_e32 v136, v136, v220
	s_wait_dscnt 0x12
	v_add_f32_e32 v137, v137, v221
	s_wait_dscnt 0x11
	v_add_f32_e32 v150, v150, v222
	s_wait_dscnt 0x10
	v_add_f32_e32 v151, v151, v223
	s_wait_dscnt 0xf
	v_add_f32_e32 v152, v152, v224
	s_wait_dscnt 0xe
	v_add_f32_e32 v153, v153, v225
	s_wait_dscnt 0xd
	v_add_f32_e32 v154, v154, v226
	s_wait_dscnt 0xc
	v_add_f32_e32 v155, v155, v227
	s_wait_dscnt 0xb
	v_add_f32_e32 v168, v168, v228
	s_wait_dscnt 0xa
	v_add_f32_e32 v169, v169, v229
	s_wait_dscnt 0x9
	v_add_f32_e32 v170, v170, v230
	s_wait_dscnt 0x8
	v_add_f32_e32 v171, v171, v231
	s_wait_dscnt 0x7
	v_add_f32_e32 v172, v172, v232
	s_wait_dscnt 0x6
	v_add_f32_e32 v173, v173, v233
	s_wait_dscnt 0x5
	v_add_f32_e32 v186, v186, v234
	s_wait_dscnt 0x4
	v_add_f32_e32 v187, v187, v235
	s_wait_dscnt 0x3
	v_add_f32_e32 v188, v188, v236
	s_wait_dscnt 0x2
	v_add_f32_e32 v189, v189, v237
	s_wait_dscnt 0x1
	v_add_f32_e32 v190, v190, v238
	s_wait_dscnt 0x0
	v_add_f32_e32 v191, v191, v239
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v72, s46
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v73, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v74, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v75, s47
	s_mul_i32 s47, s45, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v76, s47
	s_mul_i32 s47, s45, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v77, s47
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b6_p0_single
	global_load_b32 v36, v72, s[8:9] offset:0
	global_load_b32 v37, v72, s[8:9] offset:4
	global_load_b32 v42, v73, s[8:9] offset:0
	global_load_b32 v43, v73, s[8:9] offset:4
	global_load_b32 v48, v74, s[8:9] offset:0
	global_load_b32 v49, v74, s[8:9] offset:4
	global_load_b32 v54, v75, s[8:9] offset:0
	global_load_b32 v55, v75, s[8:9] offset:4
	global_load_b32 v60, v76, s[8:9] offset:0
	global_load_b32 v61, v76, s[8:9] offset:4
	global_load_b32 v66, v77, s[8:9] offset:0
	global_load_b32 v67, v77, s[8:9] offset:4
	s_wait_loadcnt 0xb
	v_add_f32_e32 v36, v96, v36
	global_store_b32 v72, v36, s[8:9] offset:0
	s_wait_loadcnt 0xa
	v_add_f32_e32 v37, v97, v37
	global_store_b32 v72, v37, s[8:9] offset:4
	s_wait_loadcnt 0x9
	v_add_f32_e32 v42, v114, v42
	global_store_b32 v73, v42, s[8:9] offset:0
	s_wait_loadcnt 0x8
	v_add_f32_e32 v43, v115, v43
	global_store_b32 v73, v43, s[8:9] offset:4
	s_wait_loadcnt 0x7
	v_add_f32_e32 v48, v132, v48
	global_store_b32 v74, v48, s[8:9] offset:0
	s_wait_loadcnt 0x6
	v_add_f32_e32 v49, v133, v49
	global_store_b32 v74, v49, s[8:9] offset:4
	s_wait_loadcnt 0x5
	v_add_f32_e32 v54, v150, v54
	global_store_b32 v75, v54, s[8:9] offset:0
	s_wait_loadcnt 0x4
	v_add_f32_e32 v55, v151, v55
	global_store_b32 v75, v55, s[8:9] offset:4
	s_wait_loadcnt 0x3
	v_add_f32_e32 v60, v168, v60
	global_store_b32 v76, v60, s[8:9] offset:0
	s_wait_loadcnt 0x2
	v_add_f32_e32 v61, v169, v61
	global_store_b32 v76, v61, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v66, v186, v66
	global_store_b32 v77, v66, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v67, v187, v67
	global_store_b32 v77, v67, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b6_p0_next
	.Lrx_b6_p0_single:
	global_load_b32 v36, v72, s[8:9] offset:0
	global_load_b32 v42, v73, s[8:9] offset:0
	global_load_b32 v48, v74, s[8:9] offset:0
	global_load_b32 v54, v75, s[8:9] offset:0
	global_load_b32 v60, v76, s[8:9] offset:0
	global_load_b32 v66, v77, s[8:9] offset:0
	s_wait_loadcnt 0x5
	v_add_f32_e32 v36, v36, v96
	global_store_b32 v72, v36, s[8:9] offset:0
	s_wait_loadcnt 0x4
	v_add_f32_e32 v42, v42, v114
	global_store_b32 v73, v42, s[8:9] offset:0
	s_wait_loadcnt 0x3
	v_add_f32_e32 v48, v48, v132
	global_store_b32 v74, v48, s[8:9] offset:0
	s_wait_loadcnt 0x2
	v_add_f32_e32 v54, v54, v150
	global_store_b32 v75, v54, s[8:9] offset:0
	s_wait_loadcnt 0x1
	v_add_f32_e32 v60, v60, v168
	global_store_b32 v76, v60, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v66, v66, v186
	global_store_b32 v77, v66, s[8:9] offset:0
	s_wait_storecnt 0x0
	s_branch .Lrx_b6_stored
	.Lrx_b6_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b6_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b6_p1_single
	global_load_b32 v38, v72, s[8:9] offset:8
	global_load_b32 v39, v72, s[8:9] offset:12
	global_load_b32 v44, v73, s[8:9] offset:8
	global_load_b32 v45, v73, s[8:9] offset:12
	global_load_b32 v50, v74, s[8:9] offset:8
	global_load_b32 v51, v74, s[8:9] offset:12
	global_load_b32 v56, v75, s[8:9] offset:8
	global_load_b32 v57, v75, s[8:9] offset:12
	global_load_b32 v62, v76, s[8:9] offset:8
	global_load_b32 v63, v76, s[8:9] offset:12
	global_load_b32 v68, v77, s[8:9] offset:8
	global_load_b32 v69, v77, s[8:9] offset:12
	s_wait_loadcnt 0xb
	v_add_f32_e32 v38, v98, v38
	global_store_b32 v72, v38, s[8:9] offset:8
	s_wait_loadcnt 0xa
	v_add_f32_e32 v39, v99, v39
	global_store_b32 v72, v39, s[8:9] offset:12
	s_wait_loadcnt 0x9
	v_add_f32_e32 v44, v116, v44
	global_store_b32 v73, v44, s[8:9] offset:8
	s_wait_loadcnt 0x8
	v_add_f32_e32 v45, v117, v45
	global_store_b32 v73, v45, s[8:9] offset:12
	s_wait_loadcnt 0x7
	v_add_f32_e32 v50, v134, v50
	global_store_b32 v74, v50, s[8:9] offset:8
	s_wait_loadcnt 0x6
	v_add_f32_e32 v51, v135, v51
	global_store_b32 v74, v51, s[8:9] offset:12
	s_wait_loadcnt 0x5
	v_add_f32_e32 v56, v152, v56
	global_store_b32 v75, v56, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v57, v153, v57
	global_store_b32 v75, v57, s[8:9] offset:12
	s_wait_loadcnt 0x3
	v_add_f32_e32 v62, v170, v62
	global_store_b32 v76, v62, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v63, v171, v63
	global_store_b32 v76, v63, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v68, v188, v68
	global_store_b32 v77, v68, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v69, v189, v69
	global_store_b32 v77, v69, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b6_p1_next
	.Lrx_b6_p1_single:
	global_load_b32 v38, v72, s[8:9] offset:8
	global_load_b32 v44, v73, s[8:9] offset:8
	global_load_b32 v50, v74, s[8:9] offset:8
	global_load_b32 v56, v75, s[8:9] offset:8
	global_load_b32 v62, v76, s[8:9] offset:8
	global_load_b32 v68, v77, s[8:9] offset:8
	s_wait_loadcnt 0x5
	v_add_f32_e32 v38, v38, v98
	global_store_b32 v72, v38, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v44, v44, v116
	global_store_b32 v73, v44, s[8:9] offset:8
	s_wait_loadcnt 0x3
	v_add_f32_e32 v50, v50, v134
	global_store_b32 v74, v50, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v56, v56, v152
	global_store_b32 v75, v56, s[8:9] offset:8
	s_wait_loadcnt 0x1
	v_add_f32_e32 v62, v62, v170
	global_store_b32 v76, v62, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v68, v68, v188
	global_store_b32 v77, v68, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b6_stored
	.Lrx_b6_p1_next:
	s_add_co_i32 s13, s12, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b6_stored
	s_add_co_i32 s13, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b6_p2_single
	global_load_b32 v40, v72, s[8:9] offset:16
	global_load_b32 v41, v72, s[8:9] offset:20
	global_load_b32 v46, v73, s[8:9] offset:16
	global_load_b32 v47, v73, s[8:9] offset:20
	global_load_b32 v52, v74, s[8:9] offset:16
	global_load_b32 v53, v74, s[8:9] offset:20
	global_load_b32 v58, v75, s[8:9] offset:16
	global_load_b32 v59, v75, s[8:9] offset:20
	global_load_b32 v64, v76, s[8:9] offset:16
	global_load_b32 v65, v76, s[8:9] offset:20
	global_load_b32 v70, v77, s[8:9] offset:16
	global_load_b32 v71, v77, s[8:9] offset:20
	s_wait_loadcnt 0xb
	v_add_f32_e32 v40, v100, v40
	global_store_b32 v72, v40, s[8:9] offset:16
	s_wait_loadcnt 0xa
	v_add_f32_e32 v41, v101, v41
	global_store_b32 v72, v41, s[8:9] offset:20
	s_wait_loadcnt 0x9
	v_add_f32_e32 v46, v118, v46
	global_store_b32 v73, v46, s[8:9] offset:16
	s_wait_loadcnt 0x8
	v_add_f32_e32 v47, v119, v47
	global_store_b32 v73, v47, s[8:9] offset:20
	s_wait_loadcnt 0x7
	v_add_f32_e32 v52, v136, v52
	global_store_b32 v74, v52, s[8:9] offset:16
	s_wait_loadcnt 0x6
	v_add_f32_e32 v53, v137, v53
	global_store_b32 v74, v53, s[8:9] offset:20
	s_wait_loadcnt 0x5
	v_add_f32_e32 v58, v154, v58
	global_store_b32 v75, v58, s[8:9] offset:16
	s_wait_loadcnt 0x4
	v_add_f32_e32 v59, v155, v59
	global_store_b32 v75, v59, s[8:9] offset:20
	s_wait_loadcnt 0x3
	v_add_f32_e32 v64, v172, v64
	global_store_b32 v76, v64, s[8:9] offset:16
	s_wait_loadcnt 0x2
	v_add_f32_e32 v65, v173, v65
	global_store_b32 v76, v65, s[8:9] offset:20
	s_wait_loadcnt 0x1
	v_add_f32_e32 v70, v190, v70
	global_store_b32 v77, v70, s[8:9] offset:16
	s_wait_loadcnt 0x0
	v_add_f32_e32 v71, v191, v71
	global_store_b32 v77, v71, s[8:9] offset:20
	s_wait_storecnt 0x0
	s_branch .Lrx_b6_stored
	.Lrx_b6_p2_single:
	global_load_b32 v40, v72, s[8:9] offset:16
	global_load_b32 v46, v73, s[8:9] offset:16
	global_load_b32 v52, v74, s[8:9] offset:16
	global_load_b32 v58, v75, s[8:9] offset:16
	global_load_b32 v64, v76, s[8:9] offset:16
	global_load_b32 v70, v77, s[8:9] offset:16
	s_wait_loadcnt 0x5
	v_add_f32_e32 v40, v40, v100
	global_store_b32 v72, v40, s[8:9] offset:16
	s_wait_loadcnt 0x4
	v_add_f32_e32 v46, v46, v118
	global_store_b32 v73, v46, s[8:9] offset:16
	s_wait_loadcnt 0x3
	v_add_f32_e32 v52, v52, v136
	global_store_b32 v74, v52, s[8:9] offset:16
	s_wait_loadcnt 0x2
	v_add_f32_e32 v58, v58, v154
	global_store_b32 v75, v58, s[8:9] offset:16
	s_wait_loadcnt 0x1
	v_add_f32_e32 v64, v64, v172
	global_store_b32 v76, v64, s[8:9] offset:16
	s_wait_loadcnt 0x0
	v_add_f32_e32 v70, v70, v190
	global_store_b32 v77, v70, s[8:9] offset:16
	s_wait_storecnt 0x0
	s_branch .Lrx_b6_stored
	.Lrx_b6_stored:
	s_branch .Lrx_end
	.Lrx_b7:
	s_mul_i32 s12, ttmp9, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s13
	v_add_nc_u32_e32 v7, s13, v1
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s0_done
	s_mov_b32 s15, 0x0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[148:151], v2, s[28:29]
	global_load_b128 v[152:155], v2, s[28:29] offset:16
	global_load_b128 v[156:159], v2, s[30:31]
	global_load_b128 v[160:163], v2, s[30:31] offset:16
	global_load_b128 v[164:167], v2, s[32:33]
	global_load_b128 v[168:171], v2, s[32:33] offset:16
	global_load_b128 v[172:175], v2, s[34:35]
	global_load_b128 v[176:179], v2, s[34:35] offset:16
	global_load_b128 v[180:183], v2, s[36:37]
	global_load_b128 v[184:187], v2, s[36:37] offset:16
	global_load_b128 v[188:191], v2, s[38:39]
	global_load_b128 v[192:195], v2, s[38:39] offset:16
	global_load_b128 v[196:199], v2, s[40:41]
	global_load_b128 v[200:203], v2, s[40:41] offset:16
	.Lrx_b7_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v24, v20, 0, 4
	v_bfe_u32 v25, v20, 4, 4
	v_bfe_u32 v26, v20, 8, 4
	v_bfe_u32 v27, v20, 12, 4
	v_bfe_u32 v28, v20, 16, 4
	v_bfe_u32 v29, v20, 20, 4
	v_bfe_u32 v30, v20, 24, 4
	v_bfe_u32 v31, v20, 28, 4
	v_cvt_f32_ubyte0_e32 v24, v24
	v_cvt_f32_ubyte0_e32 v25, v25
	v_cvt_f32_ubyte0_e32 v26, v26
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f32_ubyte0_e32 v28, v28
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f32_ubyte0_e32 v30, v30
	v_cvt_f32_ubyte0_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v34, v21, 0, 4
	v_bfe_u32 v35, v21, 4, 4
	v_bfe_u32 v36, v21, 8, 4
	v_bfe_u32 v37, v21, 12, 4
	v_bfe_u32 v38, v21, 16, 4
	v_bfe_u32 v39, v21, 20, 4
	v_bfe_u32 v40, v21, 24, 4
	v_bfe_u32 v41, v21, 28, 4
	v_cvt_f32_ubyte0_e32 v34, v34
	v_cvt_f32_ubyte0_e32 v35, v35
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v44, v22, 0, 4
	v_bfe_u32 v45, v22, 4, 4
	v_bfe_u32 v46, v22, 8, 4
	v_bfe_u32 v47, v22, 12, 4
	v_bfe_u32 v48, v22, 16, 4
	v_bfe_u32 v49, v22, 20, 4
	v_bfe_u32 v50, v22, 24, 4
	v_bfe_u32 v51, v22, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v54, v23, 0, 4
	v_bfe_u32 v55, v23, 4, 4
	v_bfe_u32 v56, v23, 8, 4
	v_bfe_u32 v57, v23, 12, 4
	v_bfe_u32 v58, v23, 16, 4
	v_bfe_u32 v59, v23, 20, 4
	v_bfe_u32 v60, v23, 24, 4
	v_bfe_u32 v61, v23, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v148 :: v_dual_mul_f32 v33, v35, v149
	v_dual_mul_f32 v42, v44, v148 :: v_dual_mul_f32 v43, v55, v149
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v156 :: v_dual_mul_f32 v53, v35, v157
	v_dual_mul_f32 v62, v44, v156 :: v_dual_mul_f32 v63, v55, v157
	v_dual_fmac_f32 v32, v25, v149 :: v_dual_fmac_f32 v33, v34, v148
	v_dual_fmac_f32 v42, v45, v149 :: v_dual_fmac_f32 v43, v54, v148
	v_dual_fmac_f32 v52, v25, v157 :: v_dual_fmac_f32 v53, v34, v156
	v_dual_fmac_f32 v62, v45, v157 :: v_dual_fmac_f32 v63, v54, v156
	v_dual_fmac_f32 v32, v26, v150 :: v_dual_fmac_f32 v33, v36, v150
	v_dual_fmac_f32 v42, v46, v150 :: v_dual_fmac_f32 v43, v56, v150
	v_dual_fmac_f32 v52, v26, v158 :: v_dual_fmac_f32 v53, v36, v158
	v_dual_fmac_f32 v62, v46, v158 :: v_dual_fmac_f32 v63, v56, v158
	v_dual_fmac_f32 v32, v27, v151 :: v_dual_fmac_f32 v33, v37, v151
	v_dual_fmac_f32 v42, v47, v151 :: v_dual_fmac_f32 v43, v57, v151
	v_dual_fmac_f32 v52, v27, v159 :: v_dual_fmac_f32 v53, v37, v159
	v_dual_fmac_f32 v62, v47, v159 :: v_dual_fmac_f32 v63, v57, v159
	v_dual_fmac_f32 v32, v28, v152 :: v_dual_fmac_f32 v33, v38, v152
	v_dual_fmac_f32 v42, v48, v152 :: v_dual_fmac_f32 v43, v58, v152
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v52, v28, v160 :: v_dual_fmac_f32 v53, v38, v160
	v_dual_fmac_f32 v62, v48, v160 :: v_dual_fmac_f32 v63, v58, v160
	v_dual_fmac_f32 v32, v29, v153 :: v_dual_fmac_f32 v33, v39, v153
	v_dual_fmac_f32 v42, v49, v153 :: v_dual_fmac_f32 v43, v59, v153
	v_dual_fmac_f32 v52, v29, v161 :: v_dual_fmac_f32 v53, v39, v161
	v_dual_fmac_f32 v62, v49, v161 :: v_dual_fmac_f32 v63, v59, v161
	v_dual_fmac_f32 v32, v30, v154 :: v_dual_fmac_f32 v33, v40, v154
	v_dual_fmac_f32 v42, v50, v154 :: v_dual_fmac_f32 v43, v60, v154
	v_dual_fmac_f32 v52, v30, v162 :: v_dual_fmac_f32 v53, v40, v162
	v_dual_fmac_f32 v62, v50, v162 :: v_dual_fmac_f32 v63, v60, v162
	v_dual_fmac_f32 v32, v31, v155 :: v_dual_fmac_f32 v33, v41, v155
	v_dual_fmac_f32 v42, v51, v155 :: v_dual_fmac_f32 v43, v61, v155
	v_dual_fmac_f32 v52, v31, v163 :: v_dual_fmac_f32 v53, v41, v163
	v_dual_fmac_f32 v62, v51, v163 :: v_dual_fmac_f32 v63, v61, v163
	global_load_b128 v[148:151], v3, s[28:29]
	global_load_b128 v[152:155], v3, s[28:29] offset:16
	global_load_b128 v[156:159], v3, s[30:31]
	global_load_b128 v[160:163], v3, s[30:31] offset:16
	v_dual_add_f32 v64, v32, v64 :: v_dual_add_f32 v65, v65, v33
	v_dual_add_f32 v66, v42, v66 :: v_dual_add_f32 v67, v67, v43
	v_dual_add_f32 v76, v52, v76 :: v_dual_add_f32 v77, v77, v53
	v_dual_add_f32 v78, v62, v78 :: v_dual_add_f32 v79, v79, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v164 :: v_dual_mul_f32 v33, v35, v165
	v_dual_mul_f32 v42, v44, v164 :: v_dual_mul_f32 v43, v55, v165
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v172 :: v_dual_mul_f32 v53, v35, v173
	v_dual_mul_f32 v62, v44, v172 :: v_dual_mul_f32 v63, v55, v173
	v_dual_fmac_f32 v32, v25, v165 :: v_dual_fmac_f32 v33, v34, v164
	v_dual_fmac_f32 v42, v45, v165 :: v_dual_fmac_f32 v43, v54, v164
	v_dual_fmac_f32 v52, v25, v173 :: v_dual_fmac_f32 v53, v34, v172
	v_dual_fmac_f32 v62, v45, v173 :: v_dual_fmac_f32 v63, v54, v172
	v_dual_fmac_f32 v32, v26, v166 :: v_dual_fmac_f32 v33, v36, v166
	v_dual_fmac_f32 v42, v46, v166 :: v_dual_fmac_f32 v43, v56, v166
	v_dual_fmac_f32 v52, v26, v174 :: v_dual_fmac_f32 v53, v36, v174
	v_dual_fmac_f32 v62, v46, v174 :: v_dual_fmac_f32 v63, v56, v174
	v_dual_fmac_f32 v32, v27, v167 :: v_dual_fmac_f32 v33, v37, v167
	v_dual_fmac_f32 v42, v47, v167 :: v_dual_fmac_f32 v43, v57, v167
	v_dual_fmac_f32 v52, v27, v175 :: v_dual_fmac_f32 v53, v37, v175
	v_dual_fmac_f32 v62, v47, v175 :: v_dual_fmac_f32 v63, v57, v175
	v_dual_fmac_f32 v32, v28, v168 :: v_dual_fmac_f32 v33, v38, v168
	v_dual_fmac_f32 v42, v48, v168 :: v_dual_fmac_f32 v43, v58, v168
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v52, v28, v176 :: v_dual_fmac_f32 v53, v38, v176
	v_dual_fmac_f32 v62, v48, v176 :: v_dual_fmac_f32 v63, v58, v176
	v_dual_fmac_f32 v32, v29, v169 :: v_dual_fmac_f32 v33, v39, v169
	v_dual_fmac_f32 v42, v49, v169 :: v_dual_fmac_f32 v43, v59, v169
	v_dual_fmac_f32 v52, v29, v177 :: v_dual_fmac_f32 v53, v39, v177
	v_dual_fmac_f32 v62, v49, v177 :: v_dual_fmac_f32 v63, v59, v177
	v_dual_fmac_f32 v32, v30, v170 :: v_dual_fmac_f32 v33, v40, v170
	v_dual_fmac_f32 v42, v50, v170 :: v_dual_fmac_f32 v43, v60, v170
	v_dual_fmac_f32 v52, v30, v178 :: v_dual_fmac_f32 v53, v40, v178
	v_dual_fmac_f32 v62, v50, v178 :: v_dual_fmac_f32 v63, v60, v178
	v_dual_fmac_f32 v32, v31, v171 :: v_dual_fmac_f32 v33, v41, v171
	v_dual_fmac_f32 v42, v51, v171 :: v_dual_fmac_f32 v43, v61, v171
	v_dual_fmac_f32 v52, v31, v179 :: v_dual_fmac_f32 v53, v41, v179
	v_dual_fmac_f32 v62, v51, v179 :: v_dual_fmac_f32 v63, v61, v179
	global_load_b128 v[164:167], v3, s[32:33]
	global_load_b128 v[168:171], v3, s[32:33] offset:16
	global_load_b128 v[172:175], v3, s[34:35]
	global_load_b128 v[176:179], v3, s[34:35] offset:16
	v_dual_add_f32 v88, v32, v88 :: v_dual_add_f32 v89, v89, v33
	v_dual_add_f32 v90, v42, v90 :: v_dual_add_f32 v91, v91, v43
	v_dual_add_f32 v100, v52, v100 :: v_dual_add_f32 v101, v101, v53
	v_dual_add_f32 v102, v62, v102 :: v_dual_add_f32 v103, v103, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v180 :: v_dual_mul_f32 v33, v35, v181
	v_dual_mul_f32 v42, v44, v180 :: v_dual_mul_f32 v43, v55, v181
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v188 :: v_dual_mul_f32 v53, v35, v189
	v_dual_mul_f32 v62, v44, v188 :: v_dual_mul_f32 v63, v55, v189
	v_dual_fmac_f32 v32, v25, v181 :: v_dual_fmac_f32 v33, v34, v180
	v_dual_fmac_f32 v42, v45, v181 :: v_dual_fmac_f32 v43, v54, v180
	v_dual_fmac_f32 v52, v25, v189 :: v_dual_fmac_f32 v53, v34, v188
	v_dual_fmac_f32 v62, v45, v189 :: v_dual_fmac_f32 v63, v54, v188
	v_dual_fmac_f32 v32, v26, v182 :: v_dual_fmac_f32 v33, v36, v182
	v_dual_fmac_f32 v42, v46, v182 :: v_dual_fmac_f32 v43, v56, v182
	v_dual_fmac_f32 v52, v26, v190 :: v_dual_fmac_f32 v53, v36, v190
	v_dual_fmac_f32 v62, v46, v190 :: v_dual_fmac_f32 v63, v56, v190
	v_dual_fmac_f32 v32, v27, v183 :: v_dual_fmac_f32 v33, v37, v183
	v_dual_fmac_f32 v42, v47, v183 :: v_dual_fmac_f32 v43, v57, v183
	v_dual_fmac_f32 v52, v27, v191 :: v_dual_fmac_f32 v53, v37, v191
	v_dual_fmac_f32 v62, v47, v191 :: v_dual_fmac_f32 v63, v57, v191
	v_dual_fmac_f32 v32, v28, v184 :: v_dual_fmac_f32 v33, v38, v184
	v_dual_fmac_f32 v42, v48, v184 :: v_dual_fmac_f32 v43, v58, v184
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v52, v28, v192 :: v_dual_fmac_f32 v53, v38, v192
	v_dual_fmac_f32 v62, v48, v192 :: v_dual_fmac_f32 v63, v58, v192
	v_dual_fmac_f32 v32, v29, v185 :: v_dual_fmac_f32 v33, v39, v185
	v_dual_fmac_f32 v42, v49, v185 :: v_dual_fmac_f32 v43, v59, v185
	v_dual_fmac_f32 v52, v29, v193 :: v_dual_fmac_f32 v53, v39, v193
	v_dual_fmac_f32 v62, v49, v193 :: v_dual_fmac_f32 v63, v59, v193
	v_dual_fmac_f32 v32, v30, v186 :: v_dual_fmac_f32 v33, v40, v186
	v_dual_fmac_f32 v42, v50, v186 :: v_dual_fmac_f32 v43, v60, v186
	v_dual_fmac_f32 v52, v30, v194 :: v_dual_fmac_f32 v53, v40, v194
	v_dual_fmac_f32 v62, v50, v194 :: v_dual_fmac_f32 v63, v60, v194
	v_dual_fmac_f32 v32, v31, v187 :: v_dual_fmac_f32 v33, v41, v187
	v_dual_fmac_f32 v42, v51, v187 :: v_dual_fmac_f32 v43, v61, v187
	v_dual_fmac_f32 v52, v31, v195 :: v_dual_fmac_f32 v53, v41, v195
	v_dual_fmac_f32 v62, v51, v195 :: v_dual_fmac_f32 v63, v61, v195
	global_load_b128 v[180:183], v3, s[36:37]
	global_load_b128 v[184:187], v3, s[36:37] offset:16
	global_load_b128 v[188:191], v3, s[38:39]
	global_load_b128 v[192:195], v3, s[38:39] offset:16
	v_dual_add_f32 v112, v32, v112 :: v_dual_add_f32 v113, v113, v33
	v_dual_add_f32 v114, v42, v114 :: v_dual_add_f32 v115, v115, v43
	v_dual_add_f32 v124, v52, v124 :: v_dual_add_f32 v125, v125, v53
	v_dual_add_f32 v126, v62, v126 :: v_dual_add_f32 v127, v127, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v196 :: v_dual_mul_f32 v33, v35, v197
	v_dual_mul_f32 v42, v44, v196 :: v_dual_mul_f32 v43, v55, v197
	v_dual_fmac_f32 v32, v25, v197 :: v_dual_fmac_f32 v33, v34, v196
	v_dual_fmac_f32 v42, v45, v197 :: v_dual_fmac_f32 v43, v54, v196
	v_dual_fmac_f32 v32, v26, v198 :: v_dual_fmac_f32 v33, v36, v198
	v_dual_fmac_f32 v42, v46, v198 :: v_dual_fmac_f32 v43, v56, v198
	v_dual_fmac_f32 v32, v27, v199 :: v_dual_fmac_f32 v33, v37, v199
	v_dual_fmac_f32 v42, v47, v199 :: v_dual_fmac_f32 v43, v57, v199
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v32, v28, v200 :: v_dual_fmac_f32 v33, v38, v200
	v_dual_fmac_f32 v42, v48, v200 :: v_dual_fmac_f32 v43, v58, v200
	v_dual_fmac_f32 v32, v29, v201 :: v_dual_fmac_f32 v33, v39, v201
	v_dual_fmac_f32 v42, v49, v201 :: v_dual_fmac_f32 v43, v59, v201
	v_dual_fmac_f32 v32, v30, v202 :: v_dual_fmac_f32 v33, v40, v202
	v_dual_fmac_f32 v42, v50, v202 :: v_dual_fmac_f32 v43, v60, v202
	v_dual_fmac_f32 v32, v31, v203 :: v_dual_fmac_f32 v33, v41, v203
	v_dual_fmac_f32 v42, v51, v203 :: v_dual_fmac_f32 v43, v61, v203
	global_load_b128 v[196:199], v3, s[40:41]
	global_load_b128 v[200:203], v3, s[40:41] offset:16
	v_dual_add_f32 v136, v32, v136 :: v_dual_add_f32 v137, v137, v33
	v_dual_add_f32 v138, v42, v138 :: v_dual_add_f32 v139, v139, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s0
	s_wait_loadcnt 0x0
	.Lrx_b7_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[148:151], v2, s[28:29]
	global_load_b128 v[152:155], v2, s[28:29] offset:16
	global_load_b128 v[156:159], v2, s[30:31]
	global_load_b128 v[160:163], v2, s[30:31] offset:16
	global_load_b128 v[164:167], v2, s[32:33]
	global_load_b128 v[168:171], v2, s[32:33] offset:16
	global_load_b128 v[172:175], v2, s[34:35]
	global_load_b128 v[176:179], v2, s[34:35] offset:16
	global_load_b128 v[180:183], v2, s[36:37]
	global_load_b128 v[184:187], v2, s[36:37] offset:16
	global_load_b128 v[188:191], v2, s[38:39]
	global_load_b128 v[192:195], v2, s[38:39] offset:16
	global_load_b128 v[196:199], v2, s[40:41]
	global_load_b128 v[200:203], v2, s[40:41] offset:16
	.Lrx_b7_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v24, v20, 0, 4
	v_bfe_u32 v25, v20, 4, 4
	v_bfe_u32 v26, v20, 8, 4
	v_bfe_u32 v27, v20, 12, 4
	v_bfe_u32 v28, v20, 16, 4
	v_bfe_u32 v29, v20, 20, 4
	v_bfe_u32 v30, v20, 24, 4
	v_bfe_u32 v31, v20, 28, 4
	v_cvt_f32_ubyte0_e32 v24, v24
	v_cvt_f32_ubyte0_e32 v25, v25
	v_cvt_f32_ubyte0_e32 v26, v26
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f32_ubyte0_e32 v28, v28
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f32_ubyte0_e32 v30, v30
	v_cvt_f32_ubyte0_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v34, v21, 0, 4
	v_bfe_u32 v35, v21, 4, 4
	v_bfe_u32 v36, v21, 8, 4
	v_bfe_u32 v37, v21, 12, 4
	v_bfe_u32 v38, v21, 16, 4
	v_bfe_u32 v39, v21, 20, 4
	v_bfe_u32 v40, v21, 24, 4
	v_bfe_u32 v41, v21, 28, 4
	v_cvt_f32_ubyte0_e32 v34, v34
	v_cvt_f32_ubyte0_e32 v35, v35
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v44, v22, 0, 4
	v_bfe_u32 v45, v22, 4, 4
	v_bfe_u32 v46, v22, 8, 4
	v_bfe_u32 v47, v22, 12, 4
	v_bfe_u32 v48, v22, 16, 4
	v_bfe_u32 v49, v22, 20, 4
	v_bfe_u32 v50, v22, 24, 4
	v_bfe_u32 v51, v22, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v54, v23, 0, 4
	v_bfe_u32 v55, v23, 4, 4
	v_bfe_u32 v56, v23, 8, 4
	v_bfe_u32 v57, v23, 12, 4
	v_bfe_u32 v58, v23, 16, 4
	v_bfe_u32 v59, v23, 20, 4
	v_bfe_u32 v60, v23, 24, 4
	v_bfe_u32 v61, v23, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v148 :: v_dual_mul_f32 v33, v35, v149
	v_dual_mul_f32 v42, v44, v148 :: v_dual_mul_f32 v43, v55, v149
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v156 :: v_dual_mul_f32 v53, v35, v157
	v_dual_mul_f32 v62, v44, v156 :: v_dual_mul_f32 v63, v55, v157
	v_dual_fmac_f32 v32, v25, v149 :: v_dual_fmac_f32 v33, v34, v148
	v_dual_fmac_f32 v42, v45, v149 :: v_dual_fmac_f32 v43, v54, v148
	v_dual_fmac_f32 v52, v25, v157 :: v_dual_fmac_f32 v53, v34, v156
	v_dual_fmac_f32 v62, v45, v157 :: v_dual_fmac_f32 v63, v54, v156
	v_dual_fmac_f32 v32, v26, v150 :: v_dual_fmac_f32 v33, v36, v150
	v_dual_fmac_f32 v42, v46, v150 :: v_dual_fmac_f32 v43, v56, v150
	v_dual_fmac_f32 v52, v26, v158 :: v_dual_fmac_f32 v53, v36, v158
	v_dual_fmac_f32 v62, v46, v158 :: v_dual_fmac_f32 v63, v56, v158
	v_dual_fmac_f32 v32, v27, v151 :: v_dual_fmac_f32 v33, v37, v151
	v_dual_fmac_f32 v42, v47, v151 :: v_dual_fmac_f32 v43, v57, v151
	v_dual_fmac_f32 v52, v27, v159 :: v_dual_fmac_f32 v53, v37, v159
	v_dual_fmac_f32 v62, v47, v159 :: v_dual_fmac_f32 v63, v57, v159
	v_dual_fmac_f32 v32, v28, v152 :: v_dual_fmac_f32 v33, v38, v152
	v_dual_fmac_f32 v42, v48, v152 :: v_dual_fmac_f32 v43, v58, v152
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v52, v28, v160 :: v_dual_fmac_f32 v53, v38, v160
	v_dual_fmac_f32 v62, v48, v160 :: v_dual_fmac_f32 v63, v58, v160
	v_dual_fmac_f32 v32, v29, v153 :: v_dual_fmac_f32 v33, v39, v153
	v_dual_fmac_f32 v42, v49, v153 :: v_dual_fmac_f32 v43, v59, v153
	v_dual_fmac_f32 v52, v29, v161 :: v_dual_fmac_f32 v53, v39, v161
	v_dual_fmac_f32 v62, v49, v161 :: v_dual_fmac_f32 v63, v59, v161
	v_dual_fmac_f32 v32, v30, v154 :: v_dual_fmac_f32 v33, v40, v154
	v_dual_fmac_f32 v42, v50, v154 :: v_dual_fmac_f32 v43, v60, v154
	v_dual_fmac_f32 v52, v30, v162 :: v_dual_fmac_f32 v53, v40, v162
	v_dual_fmac_f32 v62, v50, v162 :: v_dual_fmac_f32 v63, v60, v162
	v_dual_fmac_f32 v32, v31, v155 :: v_dual_fmac_f32 v33, v41, v155
	v_dual_fmac_f32 v42, v51, v155 :: v_dual_fmac_f32 v43, v61, v155
	v_dual_fmac_f32 v52, v31, v163 :: v_dual_fmac_f32 v53, v41, v163
	v_dual_fmac_f32 v62, v51, v163 :: v_dual_fmac_f32 v63, v61, v163
	global_load_b128 v[148:151], v3, s[28:29]
	global_load_b128 v[152:155], v3, s[28:29] offset:16
	global_load_b128 v[156:159], v3, s[30:31]
	global_load_b128 v[160:163], v3, s[30:31] offset:16
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v164 :: v_dual_mul_f32 v33, v35, v165
	v_dual_mul_f32 v42, v44, v164 :: v_dual_mul_f32 v43, v55, v165
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v172 :: v_dual_mul_f32 v53, v35, v173
	v_dual_mul_f32 v62, v44, v172 :: v_dual_mul_f32 v63, v55, v173
	v_dual_fmac_f32 v32, v25, v165 :: v_dual_fmac_f32 v33, v34, v164
	v_dual_fmac_f32 v42, v45, v165 :: v_dual_fmac_f32 v43, v54, v164
	v_dual_fmac_f32 v52, v25, v173 :: v_dual_fmac_f32 v53, v34, v172
	v_dual_fmac_f32 v62, v45, v173 :: v_dual_fmac_f32 v63, v54, v172
	v_dual_fmac_f32 v32, v26, v166 :: v_dual_fmac_f32 v33, v36, v166
	v_dual_fmac_f32 v42, v46, v166 :: v_dual_fmac_f32 v43, v56, v166
	v_dual_fmac_f32 v52, v26, v174 :: v_dual_fmac_f32 v53, v36, v174
	v_dual_fmac_f32 v62, v46, v174 :: v_dual_fmac_f32 v63, v56, v174
	v_dual_fmac_f32 v32, v27, v167 :: v_dual_fmac_f32 v33, v37, v167
	v_dual_fmac_f32 v42, v47, v167 :: v_dual_fmac_f32 v43, v57, v167
	v_dual_fmac_f32 v52, v27, v175 :: v_dual_fmac_f32 v53, v37, v175
	v_dual_fmac_f32 v62, v47, v175 :: v_dual_fmac_f32 v63, v57, v175
	v_dual_fmac_f32 v32, v28, v168 :: v_dual_fmac_f32 v33, v38, v168
	v_dual_fmac_f32 v42, v48, v168 :: v_dual_fmac_f32 v43, v58, v168
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v52, v28, v176 :: v_dual_fmac_f32 v53, v38, v176
	v_dual_fmac_f32 v62, v48, v176 :: v_dual_fmac_f32 v63, v58, v176
	v_dual_fmac_f32 v32, v29, v169 :: v_dual_fmac_f32 v33, v39, v169
	v_dual_fmac_f32 v42, v49, v169 :: v_dual_fmac_f32 v43, v59, v169
	v_dual_fmac_f32 v52, v29, v177 :: v_dual_fmac_f32 v53, v39, v177
	v_dual_fmac_f32 v62, v49, v177 :: v_dual_fmac_f32 v63, v59, v177
	v_dual_fmac_f32 v32, v30, v170 :: v_dual_fmac_f32 v33, v40, v170
	v_dual_fmac_f32 v42, v50, v170 :: v_dual_fmac_f32 v43, v60, v170
	v_dual_fmac_f32 v52, v30, v178 :: v_dual_fmac_f32 v53, v40, v178
	v_dual_fmac_f32 v62, v50, v178 :: v_dual_fmac_f32 v63, v60, v178
	v_dual_fmac_f32 v32, v31, v171 :: v_dual_fmac_f32 v33, v41, v171
	v_dual_fmac_f32 v42, v51, v171 :: v_dual_fmac_f32 v43, v61, v171
	v_dual_fmac_f32 v52, v31, v179 :: v_dual_fmac_f32 v53, v41, v179
	v_dual_fmac_f32 v62, v51, v179 :: v_dual_fmac_f32 v63, v61, v179
	global_load_b128 v[164:167], v3, s[32:33]
	global_load_b128 v[168:171], v3, s[32:33] offset:16
	global_load_b128 v[172:175], v3, s[34:35]
	global_load_b128 v[176:179], v3, s[34:35] offset:16
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	v_dual_add_f32 v104, v52, v104 :: v_dual_add_f32 v105, v105, v53
	v_dual_add_f32 v106, v62, v106 :: v_dual_add_f32 v107, v107, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v180 :: v_dual_mul_f32 v33, v35, v181
	v_dual_mul_f32 v42, v44, v180 :: v_dual_mul_f32 v43, v55, v181
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v188 :: v_dual_mul_f32 v53, v35, v189
	v_dual_mul_f32 v62, v44, v188 :: v_dual_mul_f32 v63, v55, v189
	v_dual_fmac_f32 v32, v25, v181 :: v_dual_fmac_f32 v33, v34, v180
	v_dual_fmac_f32 v42, v45, v181 :: v_dual_fmac_f32 v43, v54, v180
	v_dual_fmac_f32 v52, v25, v189 :: v_dual_fmac_f32 v53, v34, v188
	v_dual_fmac_f32 v62, v45, v189 :: v_dual_fmac_f32 v63, v54, v188
	v_dual_fmac_f32 v32, v26, v182 :: v_dual_fmac_f32 v33, v36, v182
	v_dual_fmac_f32 v42, v46, v182 :: v_dual_fmac_f32 v43, v56, v182
	v_dual_fmac_f32 v52, v26, v190 :: v_dual_fmac_f32 v53, v36, v190
	v_dual_fmac_f32 v62, v46, v190 :: v_dual_fmac_f32 v63, v56, v190
	v_dual_fmac_f32 v32, v27, v183 :: v_dual_fmac_f32 v33, v37, v183
	v_dual_fmac_f32 v42, v47, v183 :: v_dual_fmac_f32 v43, v57, v183
	v_dual_fmac_f32 v52, v27, v191 :: v_dual_fmac_f32 v53, v37, v191
	v_dual_fmac_f32 v62, v47, v191 :: v_dual_fmac_f32 v63, v57, v191
	v_dual_fmac_f32 v32, v28, v184 :: v_dual_fmac_f32 v33, v38, v184
	v_dual_fmac_f32 v42, v48, v184 :: v_dual_fmac_f32 v43, v58, v184
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v52, v28, v192 :: v_dual_fmac_f32 v53, v38, v192
	v_dual_fmac_f32 v62, v48, v192 :: v_dual_fmac_f32 v63, v58, v192
	v_dual_fmac_f32 v32, v29, v185 :: v_dual_fmac_f32 v33, v39, v185
	v_dual_fmac_f32 v42, v49, v185 :: v_dual_fmac_f32 v43, v59, v185
	v_dual_fmac_f32 v52, v29, v193 :: v_dual_fmac_f32 v53, v39, v193
	v_dual_fmac_f32 v62, v49, v193 :: v_dual_fmac_f32 v63, v59, v193
	v_dual_fmac_f32 v32, v30, v186 :: v_dual_fmac_f32 v33, v40, v186
	v_dual_fmac_f32 v42, v50, v186 :: v_dual_fmac_f32 v43, v60, v186
	v_dual_fmac_f32 v52, v30, v194 :: v_dual_fmac_f32 v53, v40, v194
	v_dual_fmac_f32 v62, v50, v194 :: v_dual_fmac_f32 v63, v60, v194
	v_dual_fmac_f32 v32, v31, v187 :: v_dual_fmac_f32 v33, v41, v187
	v_dual_fmac_f32 v42, v51, v187 :: v_dual_fmac_f32 v43, v61, v187
	v_dual_fmac_f32 v52, v31, v195 :: v_dual_fmac_f32 v53, v41, v195
	v_dual_fmac_f32 v62, v51, v195 :: v_dual_fmac_f32 v63, v61, v195
	global_load_b128 v[180:183], v3, s[36:37]
	global_load_b128 v[184:187], v3, s[36:37] offset:16
	global_load_b128 v[188:191], v3, s[38:39]
	global_load_b128 v[192:195], v3, s[38:39] offset:16
	v_dual_add_f32 v116, v32, v116 :: v_dual_add_f32 v117, v117, v33
	v_dual_add_f32 v118, v42, v118 :: v_dual_add_f32 v119, v119, v43
	v_dual_add_f32 v128, v52, v128 :: v_dual_add_f32 v129, v129, v53
	v_dual_add_f32 v130, v62, v130 :: v_dual_add_f32 v131, v131, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v196 :: v_dual_mul_f32 v33, v35, v197
	v_dual_mul_f32 v42, v44, v196 :: v_dual_mul_f32 v43, v55, v197
	v_dual_fmac_f32 v32, v25, v197 :: v_dual_fmac_f32 v33, v34, v196
	v_dual_fmac_f32 v42, v45, v197 :: v_dual_fmac_f32 v43, v54, v196
	v_dual_fmac_f32 v32, v26, v198 :: v_dual_fmac_f32 v33, v36, v198
	v_dual_fmac_f32 v42, v46, v198 :: v_dual_fmac_f32 v43, v56, v198
	v_dual_fmac_f32 v32, v27, v199 :: v_dual_fmac_f32 v33, v37, v199
	v_dual_fmac_f32 v42, v47, v199 :: v_dual_fmac_f32 v43, v57, v199
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v32, v28, v200 :: v_dual_fmac_f32 v33, v38, v200
	v_dual_fmac_f32 v42, v48, v200 :: v_dual_fmac_f32 v43, v58, v200
	v_dual_fmac_f32 v32, v29, v201 :: v_dual_fmac_f32 v33, v39, v201
	v_dual_fmac_f32 v42, v49, v201 :: v_dual_fmac_f32 v43, v59, v201
	v_dual_fmac_f32 v32, v30, v202 :: v_dual_fmac_f32 v33, v40, v202
	v_dual_fmac_f32 v42, v50, v202 :: v_dual_fmac_f32 v43, v60, v202
	v_dual_fmac_f32 v32, v31, v203 :: v_dual_fmac_f32 v33, v41, v203
	v_dual_fmac_f32 v42, v51, v203 :: v_dual_fmac_f32 v43, v61, v203
	global_load_b128 v[196:199], v3, s[40:41]
	global_load_b128 v[200:203], v3, s[40:41] offset:16
	v_dual_add_f32 v140, v32, v140 :: v_dual_add_f32 v141, v141, v33
	v_dual_add_f32 v142, v42, v142 :: v_dual_add_f32 v143, v143, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s1
	s_wait_loadcnt 0x0
	.Lrx_b7_s1_done:
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_dual_add_f32 v100, v100, v104 :: v_dual_add_f32 v101, v101, v105
	v_dual_add_f32 v102, v102, v106 :: v_dual_add_f32 v103, v103, v107
	v_dual_add_f32 v112, v112, v116 :: v_dual_add_f32 v113, v113, v117
	v_dual_add_f32 v114, v114, v118 :: v_dual_add_f32 v115, v115, v119
	v_dual_add_f32 v124, v124, v128 :: v_dual_add_f32 v125, v125, v129
	v_dual_add_f32 v126, v126, v130 :: v_dual_add_f32 v127, v127, v131
	v_dual_add_f32 v136, v136, v140 :: v_dual_add_f32 v137, v137, v141
	v_dual_add_f32 v138, v138, v142 :: v_dual_add_f32 v139, v139, v143
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[148:151], v2, s[28:29]
	global_load_b128 v[152:155], v2, s[28:29] offset:16
	global_load_b128 v[156:159], v2, s[30:31]
	global_load_b128 v[160:163], v2, s[30:31] offset:16
	global_load_b128 v[164:167], v2, s[32:33]
	global_load_b128 v[168:171], v2, s[32:33] offset:16
	global_load_b128 v[172:175], v2, s[34:35]
	global_load_b128 v[176:179], v2, s[34:35] offset:16
	global_load_b128 v[180:183], v2, s[36:37]
	global_load_b128 v[184:187], v2, s[36:37] offset:16
	global_load_b128 v[188:191], v2, s[38:39]
	global_load_b128 v[192:195], v2, s[38:39] offset:16
	global_load_b128 v[196:199], v2, s[40:41]
	global_load_b128 v[200:203], v2, s[40:41] offset:16
	.Lrx_b7_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v24, v20, 0, 4
	v_bfe_u32 v25, v20, 4, 4
	v_bfe_u32 v26, v20, 8, 4
	v_bfe_u32 v27, v20, 12, 4
	v_bfe_u32 v28, v20, 16, 4
	v_bfe_u32 v29, v20, 20, 4
	v_bfe_u32 v30, v20, 24, 4
	v_bfe_u32 v31, v20, 28, 4
	v_cvt_f32_ubyte0_e32 v24, v24
	v_cvt_f32_ubyte0_e32 v25, v25
	v_cvt_f32_ubyte0_e32 v26, v26
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f32_ubyte0_e32 v28, v28
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f32_ubyte0_e32 v30, v30
	v_cvt_f32_ubyte0_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v34, v21, 0, 4
	v_bfe_u32 v35, v21, 4, 4
	v_bfe_u32 v36, v21, 8, 4
	v_bfe_u32 v37, v21, 12, 4
	v_bfe_u32 v38, v21, 16, 4
	v_bfe_u32 v39, v21, 20, 4
	v_bfe_u32 v40, v21, 24, 4
	v_bfe_u32 v41, v21, 28, 4
	v_cvt_f32_ubyte0_e32 v34, v34
	v_cvt_f32_ubyte0_e32 v35, v35
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v44, v22, 0, 4
	v_bfe_u32 v45, v22, 4, 4
	v_bfe_u32 v46, v22, 8, 4
	v_bfe_u32 v47, v22, 12, 4
	v_bfe_u32 v48, v22, 16, 4
	v_bfe_u32 v49, v22, 20, 4
	v_bfe_u32 v50, v22, 24, 4
	v_bfe_u32 v51, v22, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v54, v23, 0, 4
	v_bfe_u32 v55, v23, 4, 4
	v_bfe_u32 v56, v23, 8, 4
	v_bfe_u32 v57, v23, 12, 4
	v_bfe_u32 v58, v23, 16, 4
	v_bfe_u32 v59, v23, 20, 4
	v_bfe_u32 v60, v23, 24, 4
	v_bfe_u32 v61, v23, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v148 :: v_dual_mul_f32 v33, v35, v149
	v_dual_mul_f32 v42, v44, v148 :: v_dual_mul_f32 v43, v55, v149
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v156 :: v_dual_mul_f32 v53, v35, v157
	v_dual_mul_f32 v62, v44, v156 :: v_dual_mul_f32 v63, v55, v157
	v_dual_fmac_f32 v32, v25, v149 :: v_dual_fmac_f32 v33, v34, v148
	v_dual_fmac_f32 v42, v45, v149 :: v_dual_fmac_f32 v43, v54, v148
	v_dual_fmac_f32 v52, v25, v157 :: v_dual_fmac_f32 v53, v34, v156
	v_dual_fmac_f32 v62, v45, v157 :: v_dual_fmac_f32 v63, v54, v156
	v_dual_fmac_f32 v32, v26, v150 :: v_dual_fmac_f32 v33, v36, v150
	v_dual_fmac_f32 v42, v46, v150 :: v_dual_fmac_f32 v43, v56, v150
	v_dual_fmac_f32 v52, v26, v158 :: v_dual_fmac_f32 v53, v36, v158
	v_dual_fmac_f32 v62, v46, v158 :: v_dual_fmac_f32 v63, v56, v158
	v_dual_fmac_f32 v32, v27, v151 :: v_dual_fmac_f32 v33, v37, v151
	v_dual_fmac_f32 v42, v47, v151 :: v_dual_fmac_f32 v43, v57, v151
	v_dual_fmac_f32 v52, v27, v159 :: v_dual_fmac_f32 v53, v37, v159
	v_dual_fmac_f32 v62, v47, v159 :: v_dual_fmac_f32 v63, v57, v159
	v_dual_fmac_f32 v32, v28, v152 :: v_dual_fmac_f32 v33, v38, v152
	v_dual_fmac_f32 v42, v48, v152 :: v_dual_fmac_f32 v43, v58, v152
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v52, v28, v160 :: v_dual_fmac_f32 v53, v38, v160
	v_dual_fmac_f32 v62, v48, v160 :: v_dual_fmac_f32 v63, v58, v160
	v_dual_fmac_f32 v32, v29, v153 :: v_dual_fmac_f32 v33, v39, v153
	v_dual_fmac_f32 v42, v49, v153 :: v_dual_fmac_f32 v43, v59, v153
	v_dual_fmac_f32 v52, v29, v161 :: v_dual_fmac_f32 v53, v39, v161
	v_dual_fmac_f32 v62, v49, v161 :: v_dual_fmac_f32 v63, v59, v161
	v_dual_fmac_f32 v32, v30, v154 :: v_dual_fmac_f32 v33, v40, v154
	v_dual_fmac_f32 v42, v50, v154 :: v_dual_fmac_f32 v43, v60, v154
	v_dual_fmac_f32 v52, v30, v162 :: v_dual_fmac_f32 v53, v40, v162
	v_dual_fmac_f32 v62, v50, v162 :: v_dual_fmac_f32 v63, v60, v162
	v_dual_fmac_f32 v32, v31, v155 :: v_dual_fmac_f32 v33, v41, v155
	v_dual_fmac_f32 v42, v51, v155 :: v_dual_fmac_f32 v43, v61, v155
	v_dual_fmac_f32 v52, v31, v163 :: v_dual_fmac_f32 v53, v41, v163
	v_dual_fmac_f32 v62, v51, v163 :: v_dual_fmac_f32 v63, v61, v163
	global_load_b128 v[148:151], v3, s[28:29]
	global_load_b128 v[152:155], v3, s[28:29] offset:16
	global_load_b128 v[156:159], v3, s[30:31]
	global_load_b128 v[160:163], v3, s[30:31] offset:16
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v164 :: v_dual_mul_f32 v33, v35, v165
	v_dual_mul_f32 v42, v44, v164 :: v_dual_mul_f32 v43, v55, v165
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v172 :: v_dual_mul_f32 v53, v35, v173
	v_dual_mul_f32 v62, v44, v172 :: v_dual_mul_f32 v63, v55, v173
	v_dual_fmac_f32 v32, v25, v165 :: v_dual_fmac_f32 v33, v34, v164
	v_dual_fmac_f32 v42, v45, v165 :: v_dual_fmac_f32 v43, v54, v164
	v_dual_fmac_f32 v52, v25, v173 :: v_dual_fmac_f32 v53, v34, v172
	v_dual_fmac_f32 v62, v45, v173 :: v_dual_fmac_f32 v63, v54, v172
	v_dual_fmac_f32 v32, v26, v166 :: v_dual_fmac_f32 v33, v36, v166
	v_dual_fmac_f32 v42, v46, v166 :: v_dual_fmac_f32 v43, v56, v166
	v_dual_fmac_f32 v52, v26, v174 :: v_dual_fmac_f32 v53, v36, v174
	v_dual_fmac_f32 v62, v46, v174 :: v_dual_fmac_f32 v63, v56, v174
	v_dual_fmac_f32 v32, v27, v167 :: v_dual_fmac_f32 v33, v37, v167
	v_dual_fmac_f32 v42, v47, v167 :: v_dual_fmac_f32 v43, v57, v167
	v_dual_fmac_f32 v52, v27, v175 :: v_dual_fmac_f32 v53, v37, v175
	v_dual_fmac_f32 v62, v47, v175 :: v_dual_fmac_f32 v63, v57, v175
	v_dual_fmac_f32 v32, v28, v168 :: v_dual_fmac_f32 v33, v38, v168
	v_dual_fmac_f32 v42, v48, v168 :: v_dual_fmac_f32 v43, v58, v168
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v52, v28, v176 :: v_dual_fmac_f32 v53, v38, v176
	v_dual_fmac_f32 v62, v48, v176 :: v_dual_fmac_f32 v63, v58, v176
	v_dual_fmac_f32 v32, v29, v169 :: v_dual_fmac_f32 v33, v39, v169
	v_dual_fmac_f32 v42, v49, v169 :: v_dual_fmac_f32 v43, v59, v169
	v_dual_fmac_f32 v52, v29, v177 :: v_dual_fmac_f32 v53, v39, v177
	v_dual_fmac_f32 v62, v49, v177 :: v_dual_fmac_f32 v63, v59, v177
	v_dual_fmac_f32 v32, v30, v170 :: v_dual_fmac_f32 v33, v40, v170
	v_dual_fmac_f32 v42, v50, v170 :: v_dual_fmac_f32 v43, v60, v170
	v_dual_fmac_f32 v52, v30, v178 :: v_dual_fmac_f32 v53, v40, v178
	v_dual_fmac_f32 v62, v50, v178 :: v_dual_fmac_f32 v63, v60, v178
	v_dual_fmac_f32 v32, v31, v171 :: v_dual_fmac_f32 v33, v41, v171
	v_dual_fmac_f32 v42, v51, v171 :: v_dual_fmac_f32 v43, v61, v171
	v_dual_fmac_f32 v52, v31, v179 :: v_dual_fmac_f32 v53, v41, v179
	v_dual_fmac_f32 v62, v51, v179 :: v_dual_fmac_f32 v63, v61, v179
	global_load_b128 v[164:167], v3, s[32:33]
	global_load_b128 v[168:171], v3, s[32:33] offset:16
	global_load_b128 v[172:175], v3, s[34:35]
	global_load_b128 v[176:179], v3, s[34:35] offset:16
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	v_dual_add_f32 v104, v52, v104 :: v_dual_add_f32 v105, v105, v53
	v_dual_add_f32 v106, v62, v106 :: v_dual_add_f32 v107, v107, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v180 :: v_dual_mul_f32 v33, v35, v181
	v_dual_mul_f32 v42, v44, v180 :: v_dual_mul_f32 v43, v55, v181
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v188 :: v_dual_mul_f32 v53, v35, v189
	v_dual_mul_f32 v62, v44, v188 :: v_dual_mul_f32 v63, v55, v189
	v_dual_fmac_f32 v32, v25, v181 :: v_dual_fmac_f32 v33, v34, v180
	v_dual_fmac_f32 v42, v45, v181 :: v_dual_fmac_f32 v43, v54, v180
	v_dual_fmac_f32 v52, v25, v189 :: v_dual_fmac_f32 v53, v34, v188
	v_dual_fmac_f32 v62, v45, v189 :: v_dual_fmac_f32 v63, v54, v188
	v_dual_fmac_f32 v32, v26, v182 :: v_dual_fmac_f32 v33, v36, v182
	v_dual_fmac_f32 v42, v46, v182 :: v_dual_fmac_f32 v43, v56, v182
	v_dual_fmac_f32 v52, v26, v190 :: v_dual_fmac_f32 v53, v36, v190
	v_dual_fmac_f32 v62, v46, v190 :: v_dual_fmac_f32 v63, v56, v190
	v_dual_fmac_f32 v32, v27, v183 :: v_dual_fmac_f32 v33, v37, v183
	v_dual_fmac_f32 v42, v47, v183 :: v_dual_fmac_f32 v43, v57, v183
	v_dual_fmac_f32 v52, v27, v191 :: v_dual_fmac_f32 v53, v37, v191
	v_dual_fmac_f32 v62, v47, v191 :: v_dual_fmac_f32 v63, v57, v191
	v_dual_fmac_f32 v32, v28, v184 :: v_dual_fmac_f32 v33, v38, v184
	v_dual_fmac_f32 v42, v48, v184 :: v_dual_fmac_f32 v43, v58, v184
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v52, v28, v192 :: v_dual_fmac_f32 v53, v38, v192
	v_dual_fmac_f32 v62, v48, v192 :: v_dual_fmac_f32 v63, v58, v192
	v_dual_fmac_f32 v32, v29, v185 :: v_dual_fmac_f32 v33, v39, v185
	v_dual_fmac_f32 v42, v49, v185 :: v_dual_fmac_f32 v43, v59, v185
	v_dual_fmac_f32 v52, v29, v193 :: v_dual_fmac_f32 v53, v39, v193
	v_dual_fmac_f32 v62, v49, v193 :: v_dual_fmac_f32 v63, v59, v193
	v_dual_fmac_f32 v32, v30, v186 :: v_dual_fmac_f32 v33, v40, v186
	v_dual_fmac_f32 v42, v50, v186 :: v_dual_fmac_f32 v43, v60, v186
	v_dual_fmac_f32 v52, v30, v194 :: v_dual_fmac_f32 v53, v40, v194
	v_dual_fmac_f32 v62, v50, v194 :: v_dual_fmac_f32 v63, v60, v194
	v_dual_fmac_f32 v32, v31, v187 :: v_dual_fmac_f32 v33, v41, v187
	v_dual_fmac_f32 v42, v51, v187 :: v_dual_fmac_f32 v43, v61, v187
	v_dual_fmac_f32 v52, v31, v195 :: v_dual_fmac_f32 v53, v41, v195
	v_dual_fmac_f32 v62, v51, v195 :: v_dual_fmac_f32 v63, v61, v195
	global_load_b128 v[180:183], v3, s[36:37]
	global_load_b128 v[184:187], v3, s[36:37] offset:16
	global_load_b128 v[188:191], v3, s[38:39]
	global_load_b128 v[192:195], v3, s[38:39] offset:16
	v_dual_add_f32 v116, v32, v116 :: v_dual_add_f32 v117, v117, v33
	v_dual_add_f32 v118, v42, v118 :: v_dual_add_f32 v119, v119, v43
	v_dual_add_f32 v128, v52, v128 :: v_dual_add_f32 v129, v129, v53
	v_dual_add_f32 v130, v62, v130 :: v_dual_add_f32 v131, v131, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v196 :: v_dual_mul_f32 v33, v35, v197
	v_dual_mul_f32 v42, v44, v196 :: v_dual_mul_f32 v43, v55, v197
	v_dual_fmac_f32 v32, v25, v197 :: v_dual_fmac_f32 v33, v34, v196
	v_dual_fmac_f32 v42, v45, v197 :: v_dual_fmac_f32 v43, v54, v196
	v_dual_fmac_f32 v32, v26, v198 :: v_dual_fmac_f32 v33, v36, v198
	v_dual_fmac_f32 v42, v46, v198 :: v_dual_fmac_f32 v43, v56, v198
	v_dual_fmac_f32 v32, v27, v199 :: v_dual_fmac_f32 v33, v37, v199
	v_dual_fmac_f32 v42, v47, v199 :: v_dual_fmac_f32 v43, v57, v199
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v32, v28, v200 :: v_dual_fmac_f32 v33, v38, v200
	v_dual_fmac_f32 v42, v48, v200 :: v_dual_fmac_f32 v43, v58, v200
	v_dual_fmac_f32 v32, v29, v201 :: v_dual_fmac_f32 v33, v39, v201
	v_dual_fmac_f32 v42, v49, v201 :: v_dual_fmac_f32 v43, v59, v201
	v_dual_fmac_f32 v32, v30, v202 :: v_dual_fmac_f32 v33, v40, v202
	v_dual_fmac_f32 v42, v50, v202 :: v_dual_fmac_f32 v43, v60, v202
	v_dual_fmac_f32 v32, v31, v203 :: v_dual_fmac_f32 v33, v41, v203
	v_dual_fmac_f32 v42, v51, v203 :: v_dual_fmac_f32 v43, v61, v203
	global_load_b128 v[196:199], v3, s[40:41]
	global_load_b128 v[200:203], v3, s[40:41] offset:16
	v_dual_add_f32 v140, v32, v140 :: v_dual_add_f32 v141, v141, v33
	v_dual_add_f32 v142, v42, v142 :: v_dual_add_f32 v143, v143, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s2
	s_wait_loadcnt 0x0
	.Lrx_b7_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[148:151], v2, s[28:29]
	global_load_b128 v[152:155], v2, s[28:29] offset:16
	global_load_b128 v[156:159], v2, s[30:31]
	global_load_b128 v[160:163], v2, s[30:31] offset:16
	global_load_b128 v[164:167], v2, s[32:33]
	global_load_b128 v[168:171], v2, s[32:33] offset:16
	global_load_b128 v[172:175], v2, s[34:35]
	global_load_b128 v[176:179], v2, s[34:35] offset:16
	global_load_b128 v[180:183], v2, s[36:37]
	global_load_b128 v[184:187], v2, s[36:37] offset:16
	global_load_b128 v[188:191], v2, s[38:39]
	global_load_b128 v[192:195], v2, s[38:39] offset:16
	global_load_b128 v[196:199], v2, s[40:41]
	global_load_b128 v[200:203], v2, s[40:41] offset:16
	.Lrx_b7_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v24, v20, 0, 4
	v_bfe_u32 v25, v20, 4, 4
	v_bfe_u32 v26, v20, 8, 4
	v_bfe_u32 v27, v20, 12, 4
	v_bfe_u32 v28, v20, 16, 4
	v_bfe_u32 v29, v20, 20, 4
	v_bfe_u32 v30, v20, 24, 4
	v_bfe_u32 v31, v20, 28, 4
	v_cvt_f32_ubyte0_e32 v24, v24
	v_cvt_f32_ubyte0_e32 v25, v25
	v_cvt_f32_ubyte0_e32 v26, v26
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f32_ubyte0_e32 v28, v28
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f32_ubyte0_e32 v30, v30
	v_cvt_f32_ubyte0_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v34, v21, 0, 4
	v_bfe_u32 v35, v21, 4, 4
	v_bfe_u32 v36, v21, 8, 4
	v_bfe_u32 v37, v21, 12, 4
	v_bfe_u32 v38, v21, 16, 4
	v_bfe_u32 v39, v21, 20, 4
	v_bfe_u32 v40, v21, 24, 4
	v_bfe_u32 v41, v21, 28, 4
	v_cvt_f32_ubyte0_e32 v34, v34
	v_cvt_f32_ubyte0_e32 v35, v35
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v44, v22, 0, 4
	v_bfe_u32 v45, v22, 4, 4
	v_bfe_u32 v46, v22, 8, 4
	v_bfe_u32 v47, v22, 12, 4
	v_bfe_u32 v48, v22, 16, 4
	v_bfe_u32 v49, v22, 20, 4
	v_bfe_u32 v50, v22, 24, 4
	v_bfe_u32 v51, v22, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xe
	v_bfe_u32 v54, v23, 0, 4
	v_bfe_u32 v55, v23, 4, 4
	v_bfe_u32 v56, v23, 8, 4
	v_bfe_u32 v57, v23, 12, 4
	v_bfe_u32 v58, v23, 16, 4
	v_bfe_u32 v59, v23, 20, 4
	v_bfe_u32 v60, v23, 24, 4
	v_bfe_u32 v61, v23, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v148 :: v_dual_mul_f32 v33, v35, v149
	v_dual_mul_f32 v42, v44, v148 :: v_dual_mul_f32 v43, v55, v149
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v156 :: v_dual_mul_f32 v53, v35, v157
	v_dual_mul_f32 v62, v44, v156 :: v_dual_mul_f32 v63, v55, v157
	v_dual_fmac_f32 v32, v25, v149 :: v_dual_fmac_f32 v33, v34, v148
	v_dual_fmac_f32 v42, v45, v149 :: v_dual_fmac_f32 v43, v54, v148
	v_dual_fmac_f32 v52, v25, v157 :: v_dual_fmac_f32 v53, v34, v156
	v_dual_fmac_f32 v62, v45, v157 :: v_dual_fmac_f32 v63, v54, v156
	v_dual_fmac_f32 v32, v26, v150 :: v_dual_fmac_f32 v33, v36, v150
	v_dual_fmac_f32 v42, v46, v150 :: v_dual_fmac_f32 v43, v56, v150
	v_dual_fmac_f32 v52, v26, v158 :: v_dual_fmac_f32 v53, v36, v158
	v_dual_fmac_f32 v62, v46, v158 :: v_dual_fmac_f32 v63, v56, v158
	v_dual_fmac_f32 v32, v27, v151 :: v_dual_fmac_f32 v33, v37, v151
	v_dual_fmac_f32 v42, v47, v151 :: v_dual_fmac_f32 v43, v57, v151
	v_dual_fmac_f32 v52, v27, v159 :: v_dual_fmac_f32 v53, v37, v159
	v_dual_fmac_f32 v62, v47, v159 :: v_dual_fmac_f32 v63, v57, v159
	v_dual_fmac_f32 v32, v28, v152 :: v_dual_fmac_f32 v33, v38, v152
	v_dual_fmac_f32 v42, v48, v152 :: v_dual_fmac_f32 v43, v58, v152
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v52, v28, v160 :: v_dual_fmac_f32 v53, v38, v160
	v_dual_fmac_f32 v62, v48, v160 :: v_dual_fmac_f32 v63, v58, v160
	v_dual_fmac_f32 v32, v29, v153 :: v_dual_fmac_f32 v33, v39, v153
	v_dual_fmac_f32 v42, v49, v153 :: v_dual_fmac_f32 v43, v59, v153
	v_dual_fmac_f32 v52, v29, v161 :: v_dual_fmac_f32 v53, v39, v161
	v_dual_fmac_f32 v62, v49, v161 :: v_dual_fmac_f32 v63, v59, v161
	v_dual_fmac_f32 v32, v30, v154 :: v_dual_fmac_f32 v33, v40, v154
	v_dual_fmac_f32 v42, v50, v154 :: v_dual_fmac_f32 v43, v60, v154
	v_dual_fmac_f32 v52, v30, v162 :: v_dual_fmac_f32 v53, v40, v162
	v_dual_fmac_f32 v62, v50, v162 :: v_dual_fmac_f32 v63, v60, v162
	v_dual_fmac_f32 v32, v31, v155 :: v_dual_fmac_f32 v33, v41, v155
	v_dual_fmac_f32 v42, v51, v155 :: v_dual_fmac_f32 v43, v61, v155
	v_dual_fmac_f32 v52, v31, v163 :: v_dual_fmac_f32 v53, v41, v163
	v_dual_fmac_f32 v62, v51, v163 :: v_dual_fmac_f32 v63, v61, v163
	global_load_b128 v[148:151], v3, s[28:29]
	global_load_b128 v[152:155], v3, s[28:29] offset:16
	global_load_b128 v[156:159], v3, s[30:31]
	global_load_b128 v[160:163], v3, s[30:31] offset:16
	v_dual_add_f32 v72, v32, v72 :: v_dual_add_f32 v73, v73, v33
	v_dual_add_f32 v74, v42, v74 :: v_dual_add_f32 v75, v75, v43
	v_dual_add_f32 v84, v52, v84 :: v_dual_add_f32 v85, v85, v53
	v_dual_add_f32 v86, v62, v86 :: v_dual_add_f32 v87, v87, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v164 :: v_dual_mul_f32 v33, v35, v165
	v_dual_mul_f32 v42, v44, v164 :: v_dual_mul_f32 v43, v55, v165
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v172 :: v_dual_mul_f32 v53, v35, v173
	v_dual_mul_f32 v62, v44, v172 :: v_dual_mul_f32 v63, v55, v173
	v_dual_fmac_f32 v32, v25, v165 :: v_dual_fmac_f32 v33, v34, v164
	v_dual_fmac_f32 v42, v45, v165 :: v_dual_fmac_f32 v43, v54, v164
	v_dual_fmac_f32 v52, v25, v173 :: v_dual_fmac_f32 v53, v34, v172
	v_dual_fmac_f32 v62, v45, v173 :: v_dual_fmac_f32 v63, v54, v172
	v_dual_fmac_f32 v32, v26, v166 :: v_dual_fmac_f32 v33, v36, v166
	v_dual_fmac_f32 v42, v46, v166 :: v_dual_fmac_f32 v43, v56, v166
	v_dual_fmac_f32 v52, v26, v174 :: v_dual_fmac_f32 v53, v36, v174
	v_dual_fmac_f32 v62, v46, v174 :: v_dual_fmac_f32 v63, v56, v174
	v_dual_fmac_f32 v32, v27, v167 :: v_dual_fmac_f32 v33, v37, v167
	v_dual_fmac_f32 v42, v47, v167 :: v_dual_fmac_f32 v43, v57, v167
	v_dual_fmac_f32 v52, v27, v175 :: v_dual_fmac_f32 v53, v37, v175
	v_dual_fmac_f32 v62, v47, v175 :: v_dual_fmac_f32 v63, v57, v175
	v_dual_fmac_f32 v32, v28, v168 :: v_dual_fmac_f32 v33, v38, v168
	v_dual_fmac_f32 v42, v48, v168 :: v_dual_fmac_f32 v43, v58, v168
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v52, v28, v176 :: v_dual_fmac_f32 v53, v38, v176
	v_dual_fmac_f32 v62, v48, v176 :: v_dual_fmac_f32 v63, v58, v176
	v_dual_fmac_f32 v32, v29, v169 :: v_dual_fmac_f32 v33, v39, v169
	v_dual_fmac_f32 v42, v49, v169 :: v_dual_fmac_f32 v43, v59, v169
	v_dual_fmac_f32 v52, v29, v177 :: v_dual_fmac_f32 v53, v39, v177
	v_dual_fmac_f32 v62, v49, v177 :: v_dual_fmac_f32 v63, v59, v177
	v_dual_fmac_f32 v32, v30, v170 :: v_dual_fmac_f32 v33, v40, v170
	v_dual_fmac_f32 v42, v50, v170 :: v_dual_fmac_f32 v43, v60, v170
	v_dual_fmac_f32 v52, v30, v178 :: v_dual_fmac_f32 v53, v40, v178
	v_dual_fmac_f32 v62, v50, v178 :: v_dual_fmac_f32 v63, v60, v178
	v_dual_fmac_f32 v32, v31, v171 :: v_dual_fmac_f32 v33, v41, v171
	v_dual_fmac_f32 v42, v51, v171 :: v_dual_fmac_f32 v43, v61, v171
	v_dual_fmac_f32 v52, v31, v179 :: v_dual_fmac_f32 v53, v41, v179
	v_dual_fmac_f32 v62, v51, v179 :: v_dual_fmac_f32 v63, v61, v179
	global_load_b128 v[164:167], v3, s[32:33]
	global_load_b128 v[168:171], v3, s[32:33] offset:16
	global_load_b128 v[172:175], v3, s[34:35]
	global_load_b128 v[176:179], v3, s[34:35] offset:16
	v_dual_add_f32 v96, v32, v96 :: v_dual_add_f32 v97, v97, v33
	v_dual_add_f32 v98, v42, v98 :: v_dual_add_f32 v99, v99, v43
	v_dual_add_f32 v108, v52, v108 :: v_dual_add_f32 v109, v109, v53
	v_dual_add_f32 v110, v62, v110 :: v_dual_add_f32 v111, v111, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v180 :: v_dual_mul_f32 v33, v35, v181
	v_dual_mul_f32 v42, v44, v180 :: v_dual_mul_f32 v43, v55, v181
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v188 :: v_dual_mul_f32 v53, v35, v189
	v_dual_mul_f32 v62, v44, v188 :: v_dual_mul_f32 v63, v55, v189
	v_dual_fmac_f32 v32, v25, v181 :: v_dual_fmac_f32 v33, v34, v180
	v_dual_fmac_f32 v42, v45, v181 :: v_dual_fmac_f32 v43, v54, v180
	v_dual_fmac_f32 v52, v25, v189 :: v_dual_fmac_f32 v53, v34, v188
	v_dual_fmac_f32 v62, v45, v189 :: v_dual_fmac_f32 v63, v54, v188
	v_dual_fmac_f32 v32, v26, v182 :: v_dual_fmac_f32 v33, v36, v182
	v_dual_fmac_f32 v42, v46, v182 :: v_dual_fmac_f32 v43, v56, v182
	v_dual_fmac_f32 v52, v26, v190 :: v_dual_fmac_f32 v53, v36, v190
	v_dual_fmac_f32 v62, v46, v190 :: v_dual_fmac_f32 v63, v56, v190
	v_dual_fmac_f32 v32, v27, v183 :: v_dual_fmac_f32 v33, v37, v183
	v_dual_fmac_f32 v42, v47, v183 :: v_dual_fmac_f32 v43, v57, v183
	v_dual_fmac_f32 v52, v27, v191 :: v_dual_fmac_f32 v53, v37, v191
	v_dual_fmac_f32 v62, v47, v191 :: v_dual_fmac_f32 v63, v57, v191
	v_dual_fmac_f32 v32, v28, v184 :: v_dual_fmac_f32 v33, v38, v184
	v_dual_fmac_f32 v42, v48, v184 :: v_dual_fmac_f32 v43, v58, v184
	s_wait_loadcnt 0x12
	v_dual_fmac_f32 v52, v28, v192 :: v_dual_fmac_f32 v53, v38, v192
	v_dual_fmac_f32 v62, v48, v192 :: v_dual_fmac_f32 v63, v58, v192
	v_dual_fmac_f32 v32, v29, v185 :: v_dual_fmac_f32 v33, v39, v185
	v_dual_fmac_f32 v42, v49, v185 :: v_dual_fmac_f32 v43, v59, v185
	v_dual_fmac_f32 v52, v29, v193 :: v_dual_fmac_f32 v53, v39, v193
	v_dual_fmac_f32 v62, v49, v193 :: v_dual_fmac_f32 v63, v59, v193
	v_dual_fmac_f32 v32, v30, v186 :: v_dual_fmac_f32 v33, v40, v186
	v_dual_fmac_f32 v42, v50, v186 :: v_dual_fmac_f32 v43, v60, v186
	v_dual_fmac_f32 v52, v30, v194 :: v_dual_fmac_f32 v53, v40, v194
	v_dual_fmac_f32 v62, v50, v194 :: v_dual_fmac_f32 v63, v60, v194
	v_dual_fmac_f32 v32, v31, v187 :: v_dual_fmac_f32 v33, v41, v187
	v_dual_fmac_f32 v42, v51, v187 :: v_dual_fmac_f32 v43, v61, v187
	v_dual_fmac_f32 v52, v31, v195 :: v_dual_fmac_f32 v53, v41, v195
	v_dual_fmac_f32 v62, v51, v195 :: v_dual_fmac_f32 v63, v61, v195
	global_load_b128 v[180:183], v3, s[36:37]
	global_load_b128 v[184:187], v3, s[36:37] offset:16
	global_load_b128 v[188:191], v3, s[38:39]
	global_load_b128 v[192:195], v3, s[38:39] offset:16
	v_dual_add_f32 v120, v32, v120 :: v_dual_add_f32 v121, v121, v33
	v_dual_add_f32 v122, v42, v122 :: v_dual_add_f32 v123, v123, v43
	v_dual_add_f32 v132, v52, v132 :: v_dual_add_f32 v133, v133, v53
	v_dual_add_f32 v134, v62, v134 :: v_dual_add_f32 v135, v135, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v196 :: v_dual_mul_f32 v33, v35, v197
	v_dual_mul_f32 v42, v44, v196 :: v_dual_mul_f32 v43, v55, v197
	v_dual_fmac_f32 v32, v25, v197 :: v_dual_fmac_f32 v33, v34, v196
	v_dual_fmac_f32 v42, v45, v197 :: v_dual_fmac_f32 v43, v54, v196
	v_dual_fmac_f32 v32, v26, v198 :: v_dual_fmac_f32 v33, v36, v198
	v_dual_fmac_f32 v42, v46, v198 :: v_dual_fmac_f32 v43, v56, v198
	v_dual_fmac_f32 v32, v27, v199 :: v_dual_fmac_f32 v33, v37, v199
	v_dual_fmac_f32 v42, v47, v199 :: v_dual_fmac_f32 v43, v57, v199
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v32, v28, v200 :: v_dual_fmac_f32 v33, v38, v200
	v_dual_fmac_f32 v42, v48, v200 :: v_dual_fmac_f32 v43, v58, v200
	v_dual_fmac_f32 v32, v29, v201 :: v_dual_fmac_f32 v33, v39, v201
	v_dual_fmac_f32 v42, v49, v201 :: v_dual_fmac_f32 v43, v59, v201
	v_dual_fmac_f32 v32, v30, v202 :: v_dual_fmac_f32 v33, v40, v202
	v_dual_fmac_f32 v42, v50, v202 :: v_dual_fmac_f32 v43, v60, v202
	v_dual_fmac_f32 v32, v31, v203 :: v_dual_fmac_f32 v33, v41, v203
	v_dual_fmac_f32 v42, v51, v203 :: v_dual_fmac_f32 v43, v61, v203
	global_load_b128 v[196:199], v3, s[40:41]
	global_load_b128 v[200:203], v3, s[40:41] offset:16
	v_dual_add_f32 v144, v32, v144 :: v_dual_add_f32 v145, v145, v33
	v_dual_add_f32 v146, v42, v146 :: v_dual_add_f32 v147, v147, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s3
	s_wait_loadcnt 0x0
	.Lrx_b7_s3_done:
	v_dual_add_f32 v68, v68, v72 :: v_dual_add_f32 v69, v69, v73
	v_dual_add_f32 v70, v70, v74 :: v_dual_add_f32 v71, v71, v75
	v_dual_add_f32 v80, v80, v84 :: v_dual_add_f32 v81, v81, v85
	v_dual_add_f32 v82, v82, v86 :: v_dual_add_f32 v83, v83, v87
	v_dual_add_f32 v92, v92, v96 :: v_dual_add_f32 v93, v93, v97
	v_dual_add_f32 v94, v94, v98 :: v_dual_add_f32 v95, v95, v99
	v_dual_add_f32 v104, v104, v108 :: v_dual_add_f32 v105, v105, v109
	v_dual_add_f32 v106, v106, v110 :: v_dual_add_f32 v107, v107, v111
	v_dual_add_f32 v116, v116, v120 :: v_dual_add_f32 v117, v117, v121
	v_dual_add_f32 v118, v118, v122 :: v_dual_add_f32 v119, v119, v123
	v_dual_add_f32 v128, v128, v132 :: v_dual_add_f32 v129, v129, v133
	v_dual_add_f32 v130, v130, v134 :: v_dual_add_f32 v131, v131, v135
	v_dual_add_f32 v140, v140, v144 :: v_dual_add_f32 v141, v141, v145
	v_dual_add_f32 v142, v142, v146 :: v_dual_add_f32 v143, v143, v147
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_dual_add_f32 v100, v100, v104 :: v_dual_add_f32 v101, v101, v105
	v_dual_add_f32 v102, v102, v106 :: v_dual_add_f32 v103, v103, v107
	v_dual_add_f32 v112, v112, v116 :: v_dual_add_f32 v113, v113, v117
	v_dual_add_f32 v114, v114, v118 :: v_dual_add_f32 v115, v115, v119
	v_dual_add_f32 v124, v124, v128 :: v_dual_add_f32 v125, v125, v129
	v_dual_add_f32 v126, v126, v130 :: v_dual_add_f32 v127, v127, v131
	v_dual_add_f32 v136, v136, v140 :: v_dual_add_f32 v137, v137, v141
	v_dual_add_f32 v138, v138, v142 :: v_dual_add_f32 v139, v139, v143
	ds_swizzle_b32 v148, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v149, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v150, v66 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v151, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v152, v76 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v153, v77 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v154, v78 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v155, v79 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v156, v88 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v157, v89 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v158, v90 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v159, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v160, v100 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v161, v101 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v162, v102 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v163, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v164, v112 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v165, v113 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v166, v114 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v167, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v168, v124 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v169, v125 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v170, v126 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v171, v127 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v172, v136 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v173, v137 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v174, v138 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v175, v139 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x1b
	v_add_f32_e32 v64, v64, v148
	s_wait_dscnt 0x1a
	v_add_f32_e32 v65, v65, v149
	s_wait_dscnt 0x19
	v_add_f32_e32 v66, v66, v150
	s_wait_dscnt 0x18
	v_add_f32_e32 v67, v67, v151
	s_wait_dscnt 0x17
	v_add_f32_e32 v76, v76, v152
	s_wait_dscnt 0x16
	v_add_f32_e32 v77, v77, v153
	s_wait_dscnt 0x15
	v_add_f32_e32 v78, v78, v154
	s_wait_dscnt 0x14
	v_add_f32_e32 v79, v79, v155
	s_wait_dscnt 0x13
	v_add_f32_e32 v88, v88, v156
	s_wait_dscnt 0x12
	v_add_f32_e32 v89, v89, v157
	s_wait_dscnt 0x11
	v_add_f32_e32 v90, v90, v158
	s_wait_dscnt 0x10
	v_add_f32_e32 v91, v91, v159
	s_wait_dscnt 0xf
	v_add_f32_e32 v100, v100, v160
	s_wait_dscnt 0xe
	v_add_f32_e32 v101, v101, v161
	s_wait_dscnt 0xd
	v_add_f32_e32 v102, v102, v162
	s_wait_dscnt 0xc
	v_add_f32_e32 v103, v103, v163
	s_wait_dscnt 0xb
	v_add_f32_e32 v112, v112, v164
	s_wait_dscnt 0xa
	v_add_f32_e32 v113, v113, v165
	s_wait_dscnt 0x9
	v_add_f32_e32 v114, v114, v166
	s_wait_dscnt 0x8
	v_add_f32_e32 v115, v115, v167
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v168
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v169
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v170
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v171
	s_wait_dscnt 0x3
	v_add_f32_e32 v136, v136, v172
	s_wait_dscnt 0x2
	v_add_f32_e32 v137, v137, v173
	s_wait_dscnt 0x1
	v_add_f32_e32 v138, v138, v174
	s_wait_dscnt 0x0
	v_add_f32_e32 v139, v139, v175
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v148, v3, v64
	ds_bpermute_b32 v149, v3, v65
	ds_bpermute_b32 v150, v3, v66
	ds_bpermute_b32 v151, v3, v67
	ds_bpermute_b32 v152, v3, v76
	ds_bpermute_b32 v153, v3, v77
	ds_bpermute_b32 v154, v3, v78
	ds_bpermute_b32 v155, v3, v79
	ds_bpermute_b32 v156, v3, v88
	ds_bpermute_b32 v157, v3, v89
	ds_bpermute_b32 v158, v3, v90
	ds_bpermute_b32 v159, v3, v91
	ds_bpermute_b32 v160, v3, v100
	ds_bpermute_b32 v161, v3, v101
	ds_bpermute_b32 v162, v3, v102
	ds_bpermute_b32 v163, v3, v103
	ds_bpermute_b32 v164, v3, v112
	ds_bpermute_b32 v165, v3, v113
	ds_bpermute_b32 v166, v3, v114
	ds_bpermute_b32 v167, v3, v115
	ds_bpermute_b32 v168, v3, v124
	ds_bpermute_b32 v169, v3, v125
	ds_bpermute_b32 v170, v3, v126
	ds_bpermute_b32 v171, v3, v127
	ds_bpermute_b32 v172, v3, v136
	ds_bpermute_b32 v173, v3, v137
	ds_bpermute_b32 v174, v3, v138
	ds_bpermute_b32 v175, v3, v139
	s_wait_dscnt 0x1b
	v_add_f32_e32 v64, v64, v148
	s_wait_dscnt 0x1a
	v_add_f32_e32 v65, v65, v149
	s_wait_dscnt 0x19
	v_add_f32_e32 v66, v66, v150
	s_wait_dscnt 0x18
	v_add_f32_e32 v67, v67, v151
	s_wait_dscnt 0x17
	v_add_f32_e32 v76, v76, v152
	s_wait_dscnt 0x16
	v_add_f32_e32 v77, v77, v153
	s_wait_dscnt 0x15
	v_add_f32_e32 v78, v78, v154
	s_wait_dscnt 0x14
	v_add_f32_e32 v79, v79, v155
	s_wait_dscnt 0x13
	v_add_f32_e32 v88, v88, v156
	s_wait_dscnt 0x12
	v_add_f32_e32 v89, v89, v157
	s_wait_dscnt 0x11
	v_add_f32_e32 v90, v90, v158
	s_wait_dscnt 0x10
	v_add_f32_e32 v91, v91, v159
	s_wait_dscnt 0xf
	v_add_f32_e32 v100, v100, v160
	s_wait_dscnt 0xe
	v_add_f32_e32 v101, v101, v161
	s_wait_dscnt 0xd
	v_add_f32_e32 v102, v102, v162
	s_wait_dscnt 0xc
	v_add_f32_e32 v103, v103, v163
	s_wait_dscnt 0xb
	v_add_f32_e32 v112, v112, v164
	s_wait_dscnt 0xa
	v_add_f32_e32 v113, v113, v165
	s_wait_dscnt 0x9
	v_add_f32_e32 v114, v114, v166
	s_wait_dscnt 0x8
	v_add_f32_e32 v115, v115, v167
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v168
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v169
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v170
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v171
	s_wait_dscnt 0x3
	v_add_f32_e32 v136, v136, v172
	s_wait_dscnt 0x2
	v_add_f32_e32 v137, v137, v173
	s_wait_dscnt 0x1
	v_add_f32_e32 v138, v138, v174
	s_wait_dscnt 0x0
	v_add_f32_e32 v139, v139, v175
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v148, v3, v64
	ds_bpermute_b32 v149, v3, v65
	ds_bpermute_b32 v150, v3, v66
	ds_bpermute_b32 v151, v3, v67
	ds_bpermute_b32 v152, v3, v76
	ds_bpermute_b32 v153, v3, v77
	ds_bpermute_b32 v154, v3, v78
	ds_bpermute_b32 v155, v3, v79
	ds_bpermute_b32 v156, v3, v88
	ds_bpermute_b32 v157, v3, v89
	ds_bpermute_b32 v158, v3, v90
	ds_bpermute_b32 v159, v3, v91
	ds_bpermute_b32 v160, v3, v100
	ds_bpermute_b32 v161, v3, v101
	ds_bpermute_b32 v162, v3, v102
	ds_bpermute_b32 v163, v3, v103
	ds_bpermute_b32 v164, v3, v112
	ds_bpermute_b32 v165, v3, v113
	ds_bpermute_b32 v166, v3, v114
	ds_bpermute_b32 v167, v3, v115
	ds_bpermute_b32 v168, v3, v124
	ds_bpermute_b32 v169, v3, v125
	ds_bpermute_b32 v170, v3, v126
	ds_bpermute_b32 v171, v3, v127
	ds_bpermute_b32 v172, v3, v136
	ds_bpermute_b32 v173, v3, v137
	ds_bpermute_b32 v174, v3, v138
	ds_bpermute_b32 v175, v3, v139
	s_wait_dscnt 0x1b
	v_add_f32_e32 v64, v64, v148
	s_wait_dscnt 0x1a
	v_add_f32_e32 v65, v65, v149
	s_wait_dscnt 0x19
	v_add_f32_e32 v66, v66, v150
	s_wait_dscnt 0x18
	v_add_f32_e32 v67, v67, v151
	s_wait_dscnt 0x17
	v_add_f32_e32 v76, v76, v152
	s_wait_dscnt 0x16
	v_add_f32_e32 v77, v77, v153
	s_wait_dscnt 0x15
	v_add_f32_e32 v78, v78, v154
	s_wait_dscnt 0x14
	v_add_f32_e32 v79, v79, v155
	s_wait_dscnt 0x13
	v_add_f32_e32 v88, v88, v156
	s_wait_dscnt 0x12
	v_add_f32_e32 v89, v89, v157
	s_wait_dscnt 0x11
	v_add_f32_e32 v90, v90, v158
	s_wait_dscnt 0x10
	v_add_f32_e32 v91, v91, v159
	s_wait_dscnt 0xf
	v_add_f32_e32 v100, v100, v160
	s_wait_dscnt 0xe
	v_add_f32_e32 v101, v101, v161
	s_wait_dscnt 0xd
	v_add_f32_e32 v102, v102, v162
	s_wait_dscnt 0xc
	v_add_f32_e32 v103, v103, v163
	s_wait_dscnt 0xb
	v_add_f32_e32 v112, v112, v164
	s_wait_dscnt 0xa
	v_add_f32_e32 v113, v113, v165
	s_wait_dscnt 0x9
	v_add_f32_e32 v114, v114, v166
	s_wait_dscnt 0x8
	v_add_f32_e32 v115, v115, v167
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v168
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v169
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v170
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v171
	s_wait_dscnt 0x3
	v_add_f32_e32 v136, v136, v172
	s_wait_dscnt 0x2
	v_add_f32_e32 v137, v137, v173
	s_wait_dscnt 0x1
	v_add_f32_e32 v138, v138, v174
	s_wait_dscnt 0x0
	v_add_f32_e32 v139, v139, v175
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v148, v3, v64
	ds_bpermute_b32 v149, v3, v65
	ds_bpermute_b32 v150, v3, v66
	ds_bpermute_b32 v151, v3, v67
	ds_bpermute_b32 v152, v3, v76
	ds_bpermute_b32 v153, v3, v77
	ds_bpermute_b32 v154, v3, v78
	ds_bpermute_b32 v155, v3, v79
	ds_bpermute_b32 v156, v3, v88
	ds_bpermute_b32 v157, v3, v89
	ds_bpermute_b32 v158, v3, v90
	ds_bpermute_b32 v159, v3, v91
	ds_bpermute_b32 v160, v3, v100
	ds_bpermute_b32 v161, v3, v101
	ds_bpermute_b32 v162, v3, v102
	ds_bpermute_b32 v163, v3, v103
	ds_bpermute_b32 v164, v3, v112
	ds_bpermute_b32 v165, v3, v113
	ds_bpermute_b32 v166, v3, v114
	ds_bpermute_b32 v167, v3, v115
	ds_bpermute_b32 v168, v3, v124
	ds_bpermute_b32 v169, v3, v125
	ds_bpermute_b32 v170, v3, v126
	ds_bpermute_b32 v171, v3, v127
	ds_bpermute_b32 v172, v3, v136
	ds_bpermute_b32 v173, v3, v137
	ds_bpermute_b32 v174, v3, v138
	ds_bpermute_b32 v175, v3, v139
	s_wait_dscnt 0x1b
	v_add_f32_e32 v64, v64, v148
	s_wait_dscnt 0x1a
	v_add_f32_e32 v65, v65, v149
	s_wait_dscnt 0x19
	v_add_f32_e32 v66, v66, v150
	s_wait_dscnt 0x18
	v_add_f32_e32 v67, v67, v151
	s_wait_dscnt 0x17
	v_add_f32_e32 v76, v76, v152
	s_wait_dscnt 0x16
	v_add_f32_e32 v77, v77, v153
	s_wait_dscnt 0x15
	v_add_f32_e32 v78, v78, v154
	s_wait_dscnt 0x14
	v_add_f32_e32 v79, v79, v155
	s_wait_dscnt 0x13
	v_add_f32_e32 v88, v88, v156
	s_wait_dscnt 0x12
	v_add_f32_e32 v89, v89, v157
	s_wait_dscnt 0x11
	v_add_f32_e32 v90, v90, v158
	s_wait_dscnt 0x10
	v_add_f32_e32 v91, v91, v159
	s_wait_dscnt 0xf
	v_add_f32_e32 v100, v100, v160
	s_wait_dscnt 0xe
	v_add_f32_e32 v101, v101, v161
	s_wait_dscnt 0xd
	v_add_f32_e32 v102, v102, v162
	s_wait_dscnt 0xc
	v_add_f32_e32 v103, v103, v163
	s_wait_dscnt 0xb
	v_add_f32_e32 v112, v112, v164
	s_wait_dscnt 0xa
	v_add_f32_e32 v113, v113, v165
	s_wait_dscnt 0x9
	v_add_f32_e32 v114, v114, v166
	s_wait_dscnt 0x8
	v_add_f32_e32 v115, v115, v167
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v168
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v169
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v170
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v171
	s_wait_dscnt 0x3
	v_add_f32_e32 v136, v136, v172
	s_wait_dscnt 0x2
	v_add_f32_e32 v137, v137, v173
	s_wait_dscnt 0x1
	v_add_f32_e32 v138, v138, v174
	s_wait_dscnt 0x0
	v_add_f32_e32 v139, v139, v175
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v148, v3, v64
	ds_bpermute_b32 v149, v3, v65
	ds_bpermute_b32 v150, v3, v66
	ds_bpermute_b32 v151, v3, v67
	ds_bpermute_b32 v152, v3, v76
	ds_bpermute_b32 v153, v3, v77
	ds_bpermute_b32 v154, v3, v78
	ds_bpermute_b32 v155, v3, v79
	ds_bpermute_b32 v156, v3, v88
	ds_bpermute_b32 v157, v3, v89
	ds_bpermute_b32 v158, v3, v90
	ds_bpermute_b32 v159, v3, v91
	ds_bpermute_b32 v160, v3, v100
	ds_bpermute_b32 v161, v3, v101
	ds_bpermute_b32 v162, v3, v102
	ds_bpermute_b32 v163, v3, v103
	ds_bpermute_b32 v164, v3, v112
	ds_bpermute_b32 v165, v3, v113
	ds_bpermute_b32 v166, v3, v114
	ds_bpermute_b32 v167, v3, v115
	ds_bpermute_b32 v168, v3, v124
	ds_bpermute_b32 v169, v3, v125
	ds_bpermute_b32 v170, v3, v126
	ds_bpermute_b32 v171, v3, v127
	ds_bpermute_b32 v172, v3, v136
	ds_bpermute_b32 v173, v3, v137
	ds_bpermute_b32 v174, v3, v138
	ds_bpermute_b32 v175, v3, v139
	s_wait_dscnt 0x1b
	v_add_f32_e32 v64, v64, v148
	s_wait_dscnt 0x1a
	v_add_f32_e32 v65, v65, v149
	s_wait_dscnt 0x19
	v_add_f32_e32 v66, v66, v150
	s_wait_dscnt 0x18
	v_add_f32_e32 v67, v67, v151
	s_wait_dscnt 0x17
	v_add_f32_e32 v76, v76, v152
	s_wait_dscnt 0x16
	v_add_f32_e32 v77, v77, v153
	s_wait_dscnt 0x15
	v_add_f32_e32 v78, v78, v154
	s_wait_dscnt 0x14
	v_add_f32_e32 v79, v79, v155
	s_wait_dscnt 0x13
	v_add_f32_e32 v88, v88, v156
	s_wait_dscnt 0x12
	v_add_f32_e32 v89, v89, v157
	s_wait_dscnt 0x11
	v_add_f32_e32 v90, v90, v158
	s_wait_dscnt 0x10
	v_add_f32_e32 v91, v91, v159
	s_wait_dscnt 0xf
	v_add_f32_e32 v100, v100, v160
	s_wait_dscnt 0xe
	v_add_f32_e32 v101, v101, v161
	s_wait_dscnt 0xd
	v_add_f32_e32 v102, v102, v162
	s_wait_dscnt 0xc
	v_add_f32_e32 v103, v103, v163
	s_wait_dscnt 0xb
	v_add_f32_e32 v112, v112, v164
	s_wait_dscnt 0xa
	v_add_f32_e32 v113, v113, v165
	s_wait_dscnt 0x9
	v_add_f32_e32 v114, v114, v166
	s_wait_dscnt 0x8
	v_add_f32_e32 v115, v115, v167
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v168
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v169
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v170
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v171
	s_wait_dscnt 0x3
	v_add_f32_e32 v136, v136, v172
	s_wait_dscnt 0x2
	v_add_f32_e32 v137, v137, v173
	s_wait_dscnt 0x1
	v_add_f32_e32 v138, v138, v174
	s_wait_dscnt 0x0
	v_add_f32_e32 v139, v139, v175
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v52, s46
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v53, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v54, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v55, s47
	s_mul_i32 s47, s45, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v56, s47
	s_mul_i32 s47, s45, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v57, s47
	s_mul_i32 s47, s45, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v58, s47
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b7_p0_single
	global_load_b32 v24, v52, s[8:9] offset:0
	global_load_b32 v25, v52, s[8:9] offset:4
	global_load_b32 v28, v53, s[8:9] offset:0
	global_load_b32 v29, v53, s[8:9] offset:4
	global_load_b32 v32, v54, s[8:9] offset:0
	global_load_b32 v33, v54, s[8:9] offset:4
	global_load_b32 v36, v55, s[8:9] offset:0
	global_load_b32 v37, v55, s[8:9] offset:4
	global_load_b32 v40, v56, s[8:9] offset:0
	global_load_b32 v41, v56, s[8:9] offset:4
	global_load_b32 v44, v57, s[8:9] offset:0
	global_load_b32 v45, v57, s[8:9] offset:4
	global_load_b32 v48, v58, s[8:9] offset:0
	global_load_b32 v49, v58, s[8:9] offset:4
	s_wait_loadcnt 0xd
	v_add_f32_e32 v24, v64, v24
	global_store_b32 v52, v24, s[8:9] offset:0
	s_wait_loadcnt 0xc
	v_add_f32_e32 v25, v65, v25
	global_store_b32 v52, v25, s[8:9] offset:4
	s_wait_loadcnt 0xb
	v_add_f32_e32 v28, v76, v28
	global_store_b32 v53, v28, s[8:9] offset:0
	s_wait_loadcnt 0xa
	v_add_f32_e32 v29, v77, v29
	global_store_b32 v53, v29, s[8:9] offset:4
	s_wait_loadcnt 0x9
	v_add_f32_e32 v32, v88, v32
	global_store_b32 v54, v32, s[8:9] offset:0
	s_wait_loadcnt 0x8
	v_add_f32_e32 v33, v89, v33
	global_store_b32 v54, v33, s[8:9] offset:4
	s_wait_loadcnt 0x7
	v_add_f32_e32 v36, v100, v36
	global_store_b32 v55, v36, s[8:9] offset:0
	s_wait_loadcnt 0x6
	v_add_f32_e32 v37, v101, v37
	global_store_b32 v55, v37, s[8:9] offset:4
	s_wait_loadcnt 0x5
	v_add_f32_e32 v40, v112, v40
	global_store_b32 v56, v40, s[8:9] offset:0
	s_wait_loadcnt 0x4
	v_add_f32_e32 v41, v113, v41
	global_store_b32 v56, v41, s[8:9] offset:4
	s_wait_loadcnt 0x3
	v_add_f32_e32 v44, v124, v44
	global_store_b32 v57, v44, s[8:9] offset:0
	s_wait_loadcnt 0x2
	v_add_f32_e32 v45, v125, v45
	global_store_b32 v57, v45, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v48, v136, v48
	global_store_b32 v58, v48, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v49, v137, v49
	global_store_b32 v58, v49, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b7_p0_next
	.Lrx_b7_p0_single:
	global_load_b32 v24, v52, s[8:9] offset:0
	global_load_b32 v28, v53, s[8:9] offset:0
	global_load_b32 v32, v54, s[8:9] offset:0
	global_load_b32 v36, v55, s[8:9] offset:0
	global_load_b32 v40, v56, s[8:9] offset:0
	global_load_b32 v44, v57, s[8:9] offset:0
	global_load_b32 v48, v58, s[8:9] offset:0
	s_wait_loadcnt 0x6
	v_add_f32_e32 v24, v24, v64
	global_store_b32 v52, v24, s[8:9] offset:0
	s_wait_loadcnt 0x5
	v_add_f32_e32 v28, v28, v76
	global_store_b32 v53, v28, s[8:9] offset:0
	s_wait_loadcnt 0x4
	v_add_f32_e32 v32, v32, v88
	global_store_b32 v54, v32, s[8:9] offset:0
	s_wait_loadcnt 0x3
	v_add_f32_e32 v36, v36, v100
	global_store_b32 v55, v36, s[8:9] offset:0
	s_wait_loadcnt 0x2
	v_add_f32_e32 v40, v40, v112
	global_store_b32 v56, v40, s[8:9] offset:0
	s_wait_loadcnt 0x1
	v_add_f32_e32 v44, v44, v124
	global_store_b32 v57, v44, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v48, v48, v136
	global_store_b32 v58, v48, s[8:9] offset:0
	s_wait_storecnt 0x0
	s_branch .Lrx_b7_stored
	.Lrx_b7_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b7_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b7_p1_single
	global_load_b32 v26, v52, s[8:9] offset:8
	global_load_b32 v27, v52, s[8:9] offset:12
	global_load_b32 v30, v53, s[8:9] offset:8
	global_load_b32 v31, v53, s[8:9] offset:12
	global_load_b32 v34, v54, s[8:9] offset:8
	global_load_b32 v35, v54, s[8:9] offset:12
	global_load_b32 v38, v55, s[8:9] offset:8
	global_load_b32 v39, v55, s[8:9] offset:12
	global_load_b32 v42, v56, s[8:9] offset:8
	global_load_b32 v43, v56, s[8:9] offset:12
	global_load_b32 v46, v57, s[8:9] offset:8
	global_load_b32 v47, v57, s[8:9] offset:12
	global_load_b32 v50, v58, s[8:9] offset:8
	global_load_b32 v51, v58, s[8:9] offset:12
	s_wait_loadcnt 0xd
	v_add_f32_e32 v26, v66, v26
	global_store_b32 v52, v26, s[8:9] offset:8
	s_wait_loadcnt 0xc
	v_add_f32_e32 v27, v67, v27
	global_store_b32 v52, v27, s[8:9] offset:12
	s_wait_loadcnt 0xb
	v_add_f32_e32 v30, v78, v30
	global_store_b32 v53, v30, s[8:9] offset:8
	s_wait_loadcnt 0xa
	v_add_f32_e32 v31, v79, v31
	global_store_b32 v53, v31, s[8:9] offset:12
	s_wait_loadcnt 0x9
	v_add_f32_e32 v34, v90, v34
	global_store_b32 v54, v34, s[8:9] offset:8
	s_wait_loadcnt 0x8
	v_add_f32_e32 v35, v91, v35
	global_store_b32 v54, v35, s[8:9] offset:12
	s_wait_loadcnt 0x7
	v_add_f32_e32 v38, v102, v38
	global_store_b32 v55, v38, s[8:9] offset:8
	s_wait_loadcnt 0x6
	v_add_f32_e32 v39, v103, v39
	global_store_b32 v55, v39, s[8:9] offset:12
	s_wait_loadcnt 0x5
	v_add_f32_e32 v42, v114, v42
	global_store_b32 v56, v42, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v43, v115, v43
	global_store_b32 v56, v43, s[8:9] offset:12
	s_wait_loadcnt 0x3
	v_add_f32_e32 v46, v126, v46
	global_store_b32 v57, v46, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v47, v127, v47
	global_store_b32 v57, v47, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v50, v138, v50
	global_store_b32 v58, v50, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v51, v139, v51
	global_store_b32 v58, v51, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b7_stored
	.Lrx_b7_p1_single:
	global_load_b32 v26, v52, s[8:9] offset:8
	global_load_b32 v30, v53, s[8:9] offset:8
	global_load_b32 v34, v54, s[8:9] offset:8
	global_load_b32 v38, v55, s[8:9] offset:8
	global_load_b32 v42, v56, s[8:9] offset:8
	global_load_b32 v46, v57, s[8:9] offset:8
	global_load_b32 v50, v58, s[8:9] offset:8
	s_wait_loadcnt 0x6
	v_add_f32_e32 v26, v26, v66
	global_store_b32 v52, v26, s[8:9] offset:8
	s_wait_loadcnt 0x5
	v_add_f32_e32 v30, v30, v78
	global_store_b32 v53, v30, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v34, v34, v90
	global_store_b32 v54, v34, s[8:9] offset:8
	s_wait_loadcnt 0x3
	v_add_f32_e32 v38, v38, v102
	global_store_b32 v55, v38, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v42, v42, v114
	global_store_b32 v56, v42, s[8:9] offset:8
	s_wait_loadcnt 0x1
	v_add_f32_e32 v46, v46, v126
	global_store_b32 v57, v46, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v50, v50, v138
	global_store_b32 v58, v50, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b7_stored
	.Lrx_b7_stored:
	s_branch .Lrx_end
	.Lrx_b8:
	s_mul_i32 s12, ttmp9, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s13
	v_add_nc_u32_e32 v7, s13, v1
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s0_done
	s_mov_b32 s15, 0x0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[160:163], v2, s[28:29]
	global_load_b128 v[164:167], v2, s[28:29] offset:16
	global_load_b128 v[168:171], v2, s[30:31]
	global_load_b128 v[172:175], v2, s[30:31] offset:16
	global_load_b128 v[176:179], v2, s[32:33]
	global_load_b128 v[180:183], v2, s[32:33] offset:16
	global_load_b128 v[184:187], v2, s[34:35]
	global_load_b128 v[188:191], v2, s[34:35] offset:16
	global_load_b128 v[192:195], v2, s[36:37]
	global_load_b128 v[196:199], v2, s[36:37] offset:16
	global_load_b128 v[200:203], v2, s[38:39]
	global_load_b128 v[204:207], v2, s[38:39] offset:16
	global_load_b128 v[208:211], v2, s[40:41]
	global_load_b128 v[212:215], v2, s[40:41] offset:16
	global_load_b128 v[216:219], v2, s[42:43]
	global_load_b128 v[220:223], v2, s[42:43] offset:16
	.Lrx_b8_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x16
	v_bfe_u32 v24, v20, 0, 4
	v_bfe_u32 v25, v20, 4, 4
	v_bfe_u32 v26, v20, 8, 4
	v_bfe_u32 v27, v20, 12, 4
	v_bfe_u32 v28, v20, 16, 4
	v_bfe_u32 v29, v20, 20, 4
	v_bfe_u32 v30, v20, 24, 4
	v_bfe_u32 v31, v20, 28, 4
	v_cvt_f32_ubyte0_e32 v24, v24
	v_cvt_f32_ubyte0_e32 v25, v25
	v_cvt_f32_ubyte0_e32 v26, v26
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f32_ubyte0_e32 v28, v28
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f32_ubyte0_e32 v30, v30
	v_cvt_f32_ubyte0_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v34, v21, 0, 4
	v_bfe_u32 v35, v21, 4, 4
	v_bfe_u32 v36, v21, 8, 4
	v_bfe_u32 v37, v21, 12, 4
	v_bfe_u32 v38, v21, 16, 4
	v_bfe_u32 v39, v21, 20, 4
	v_bfe_u32 v40, v21, 24, 4
	v_bfe_u32 v41, v21, 28, 4
	v_cvt_f32_ubyte0_e32 v34, v34
	v_cvt_f32_ubyte0_e32 v35, v35
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v44, v22, 0, 4
	v_bfe_u32 v45, v22, 4, 4
	v_bfe_u32 v46, v22, 8, 4
	v_bfe_u32 v47, v22, 12, 4
	v_bfe_u32 v48, v22, 16, 4
	v_bfe_u32 v49, v22, 20, 4
	v_bfe_u32 v50, v22, 24, 4
	v_bfe_u32 v51, v22, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v54, v23, 0, 4
	v_bfe_u32 v55, v23, 4, 4
	v_bfe_u32 v56, v23, 8, 4
	v_bfe_u32 v57, v23, 12, 4
	v_bfe_u32 v58, v23, 16, 4
	v_bfe_u32 v59, v23, 20, 4
	v_bfe_u32 v60, v23, 24, 4
	v_bfe_u32 v61, v23, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v160 :: v_dual_mul_f32 v33, v35, v161
	v_dual_mul_f32 v42, v44, v160 :: v_dual_mul_f32 v43, v55, v161
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v168 :: v_dual_mul_f32 v53, v35, v169
	v_dual_mul_f32 v62, v44, v168 :: v_dual_mul_f32 v63, v55, v169
	v_dual_fmac_f32 v32, v25, v161 :: v_dual_fmac_f32 v33, v34, v160
	v_dual_fmac_f32 v42, v45, v161 :: v_dual_fmac_f32 v43, v54, v160
	v_dual_fmac_f32 v52, v25, v169 :: v_dual_fmac_f32 v53, v34, v168
	v_dual_fmac_f32 v62, v45, v169 :: v_dual_fmac_f32 v63, v54, v168
	v_dual_fmac_f32 v32, v26, v162 :: v_dual_fmac_f32 v33, v36, v162
	v_dual_fmac_f32 v42, v46, v162 :: v_dual_fmac_f32 v43, v56, v162
	v_dual_fmac_f32 v52, v26, v170 :: v_dual_fmac_f32 v53, v36, v170
	v_dual_fmac_f32 v62, v46, v170 :: v_dual_fmac_f32 v63, v56, v170
	v_dual_fmac_f32 v32, v27, v163 :: v_dual_fmac_f32 v33, v37, v163
	v_dual_fmac_f32 v42, v47, v163 :: v_dual_fmac_f32 v43, v57, v163
	v_dual_fmac_f32 v52, v27, v171 :: v_dual_fmac_f32 v53, v37, v171
	v_dual_fmac_f32 v62, v47, v171 :: v_dual_fmac_f32 v63, v57, v171
	v_dual_fmac_f32 v32, v28, v164 :: v_dual_fmac_f32 v33, v38, v164
	v_dual_fmac_f32 v42, v48, v164 :: v_dual_fmac_f32 v43, v58, v164
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v172 :: v_dual_fmac_f32 v53, v38, v172
	v_dual_fmac_f32 v62, v48, v172 :: v_dual_fmac_f32 v63, v58, v172
	v_dual_fmac_f32 v32, v29, v165 :: v_dual_fmac_f32 v33, v39, v165
	v_dual_fmac_f32 v42, v49, v165 :: v_dual_fmac_f32 v43, v59, v165
	v_dual_fmac_f32 v52, v29, v173 :: v_dual_fmac_f32 v53, v39, v173
	v_dual_fmac_f32 v62, v49, v173 :: v_dual_fmac_f32 v63, v59, v173
	v_dual_fmac_f32 v32, v30, v166 :: v_dual_fmac_f32 v33, v40, v166
	v_dual_fmac_f32 v42, v50, v166 :: v_dual_fmac_f32 v43, v60, v166
	v_dual_fmac_f32 v52, v30, v174 :: v_dual_fmac_f32 v53, v40, v174
	v_dual_fmac_f32 v62, v50, v174 :: v_dual_fmac_f32 v63, v60, v174
	v_dual_fmac_f32 v32, v31, v167 :: v_dual_fmac_f32 v33, v41, v167
	v_dual_fmac_f32 v42, v51, v167 :: v_dual_fmac_f32 v43, v61, v167
	v_dual_fmac_f32 v52, v31, v175 :: v_dual_fmac_f32 v53, v41, v175
	v_dual_fmac_f32 v62, v51, v175 :: v_dual_fmac_f32 v63, v61, v175
	global_load_b128 v[160:163], v3, s[28:29]
	global_load_b128 v[164:167], v3, s[28:29] offset:16
	global_load_b128 v[168:171], v3, s[30:31]
	global_load_b128 v[172:175], v3, s[30:31] offset:16
	v_dual_add_f32 v64, v32, v64 :: v_dual_add_f32 v65, v65, v33
	v_dual_add_f32 v66, v42, v66 :: v_dual_add_f32 v67, v67, v43
	v_dual_add_f32 v76, v52, v76 :: v_dual_add_f32 v77, v77, v53
	v_dual_add_f32 v78, v62, v78 :: v_dual_add_f32 v79, v79, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v176 :: v_dual_mul_f32 v33, v35, v177
	v_dual_mul_f32 v42, v44, v176 :: v_dual_mul_f32 v43, v55, v177
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v184 :: v_dual_mul_f32 v53, v35, v185
	v_dual_mul_f32 v62, v44, v184 :: v_dual_mul_f32 v63, v55, v185
	v_dual_fmac_f32 v32, v25, v177 :: v_dual_fmac_f32 v33, v34, v176
	v_dual_fmac_f32 v42, v45, v177 :: v_dual_fmac_f32 v43, v54, v176
	v_dual_fmac_f32 v52, v25, v185 :: v_dual_fmac_f32 v53, v34, v184
	v_dual_fmac_f32 v62, v45, v185 :: v_dual_fmac_f32 v63, v54, v184
	v_dual_fmac_f32 v32, v26, v178 :: v_dual_fmac_f32 v33, v36, v178
	v_dual_fmac_f32 v42, v46, v178 :: v_dual_fmac_f32 v43, v56, v178
	v_dual_fmac_f32 v52, v26, v186 :: v_dual_fmac_f32 v53, v36, v186
	v_dual_fmac_f32 v62, v46, v186 :: v_dual_fmac_f32 v63, v56, v186
	v_dual_fmac_f32 v32, v27, v179 :: v_dual_fmac_f32 v33, v37, v179
	v_dual_fmac_f32 v42, v47, v179 :: v_dual_fmac_f32 v43, v57, v179
	v_dual_fmac_f32 v52, v27, v187 :: v_dual_fmac_f32 v53, v37, v187
	v_dual_fmac_f32 v62, v47, v187 :: v_dual_fmac_f32 v63, v57, v187
	v_dual_fmac_f32 v32, v28, v180 :: v_dual_fmac_f32 v33, v38, v180
	v_dual_fmac_f32 v42, v48, v180 :: v_dual_fmac_f32 v43, v58, v180
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v188 :: v_dual_fmac_f32 v53, v38, v188
	v_dual_fmac_f32 v62, v48, v188 :: v_dual_fmac_f32 v63, v58, v188
	v_dual_fmac_f32 v32, v29, v181 :: v_dual_fmac_f32 v33, v39, v181
	v_dual_fmac_f32 v42, v49, v181 :: v_dual_fmac_f32 v43, v59, v181
	v_dual_fmac_f32 v52, v29, v189 :: v_dual_fmac_f32 v53, v39, v189
	v_dual_fmac_f32 v62, v49, v189 :: v_dual_fmac_f32 v63, v59, v189
	v_dual_fmac_f32 v32, v30, v182 :: v_dual_fmac_f32 v33, v40, v182
	v_dual_fmac_f32 v42, v50, v182 :: v_dual_fmac_f32 v43, v60, v182
	v_dual_fmac_f32 v52, v30, v190 :: v_dual_fmac_f32 v53, v40, v190
	v_dual_fmac_f32 v62, v50, v190 :: v_dual_fmac_f32 v63, v60, v190
	v_dual_fmac_f32 v32, v31, v183 :: v_dual_fmac_f32 v33, v41, v183
	v_dual_fmac_f32 v42, v51, v183 :: v_dual_fmac_f32 v43, v61, v183
	v_dual_fmac_f32 v52, v31, v191 :: v_dual_fmac_f32 v53, v41, v191
	v_dual_fmac_f32 v62, v51, v191 :: v_dual_fmac_f32 v63, v61, v191
	global_load_b128 v[176:179], v3, s[32:33]
	global_load_b128 v[180:183], v3, s[32:33] offset:16
	global_load_b128 v[184:187], v3, s[34:35]
	global_load_b128 v[188:191], v3, s[34:35] offset:16
	v_dual_add_f32 v88, v32, v88 :: v_dual_add_f32 v89, v89, v33
	v_dual_add_f32 v90, v42, v90 :: v_dual_add_f32 v91, v91, v43
	v_dual_add_f32 v100, v52, v100 :: v_dual_add_f32 v101, v101, v53
	v_dual_add_f32 v102, v62, v102 :: v_dual_add_f32 v103, v103, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v192 :: v_dual_mul_f32 v33, v35, v193
	v_dual_mul_f32 v42, v44, v192 :: v_dual_mul_f32 v43, v55, v193
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v200 :: v_dual_mul_f32 v53, v35, v201
	v_dual_mul_f32 v62, v44, v200 :: v_dual_mul_f32 v63, v55, v201
	v_dual_fmac_f32 v32, v25, v193 :: v_dual_fmac_f32 v33, v34, v192
	v_dual_fmac_f32 v42, v45, v193 :: v_dual_fmac_f32 v43, v54, v192
	v_dual_fmac_f32 v52, v25, v201 :: v_dual_fmac_f32 v53, v34, v200
	v_dual_fmac_f32 v62, v45, v201 :: v_dual_fmac_f32 v63, v54, v200
	v_dual_fmac_f32 v32, v26, v194 :: v_dual_fmac_f32 v33, v36, v194
	v_dual_fmac_f32 v42, v46, v194 :: v_dual_fmac_f32 v43, v56, v194
	v_dual_fmac_f32 v52, v26, v202 :: v_dual_fmac_f32 v53, v36, v202
	v_dual_fmac_f32 v62, v46, v202 :: v_dual_fmac_f32 v63, v56, v202
	v_dual_fmac_f32 v32, v27, v195 :: v_dual_fmac_f32 v33, v37, v195
	v_dual_fmac_f32 v42, v47, v195 :: v_dual_fmac_f32 v43, v57, v195
	v_dual_fmac_f32 v52, v27, v203 :: v_dual_fmac_f32 v53, v37, v203
	v_dual_fmac_f32 v62, v47, v203 :: v_dual_fmac_f32 v63, v57, v203
	v_dual_fmac_f32 v32, v28, v196 :: v_dual_fmac_f32 v33, v38, v196
	v_dual_fmac_f32 v42, v48, v196 :: v_dual_fmac_f32 v43, v58, v196
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v204 :: v_dual_fmac_f32 v53, v38, v204
	v_dual_fmac_f32 v62, v48, v204 :: v_dual_fmac_f32 v63, v58, v204
	v_dual_fmac_f32 v32, v29, v197 :: v_dual_fmac_f32 v33, v39, v197
	v_dual_fmac_f32 v42, v49, v197 :: v_dual_fmac_f32 v43, v59, v197
	v_dual_fmac_f32 v52, v29, v205 :: v_dual_fmac_f32 v53, v39, v205
	v_dual_fmac_f32 v62, v49, v205 :: v_dual_fmac_f32 v63, v59, v205
	v_dual_fmac_f32 v32, v30, v198 :: v_dual_fmac_f32 v33, v40, v198
	v_dual_fmac_f32 v42, v50, v198 :: v_dual_fmac_f32 v43, v60, v198
	v_dual_fmac_f32 v52, v30, v206 :: v_dual_fmac_f32 v53, v40, v206
	v_dual_fmac_f32 v62, v50, v206 :: v_dual_fmac_f32 v63, v60, v206
	v_dual_fmac_f32 v32, v31, v199 :: v_dual_fmac_f32 v33, v41, v199
	v_dual_fmac_f32 v42, v51, v199 :: v_dual_fmac_f32 v43, v61, v199
	v_dual_fmac_f32 v52, v31, v207 :: v_dual_fmac_f32 v53, v41, v207
	v_dual_fmac_f32 v62, v51, v207 :: v_dual_fmac_f32 v63, v61, v207
	global_load_b128 v[192:195], v3, s[36:37]
	global_load_b128 v[196:199], v3, s[36:37] offset:16
	global_load_b128 v[200:203], v3, s[38:39]
	global_load_b128 v[204:207], v3, s[38:39] offset:16
	v_dual_add_f32 v112, v32, v112 :: v_dual_add_f32 v113, v113, v33
	v_dual_add_f32 v114, v42, v114 :: v_dual_add_f32 v115, v115, v43
	v_dual_add_f32 v124, v52, v124 :: v_dual_add_f32 v125, v125, v53
	v_dual_add_f32 v126, v62, v126 :: v_dual_add_f32 v127, v127, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v208 :: v_dual_mul_f32 v33, v35, v209
	v_dual_mul_f32 v42, v44, v208 :: v_dual_mul_f32 v43, v55, v209
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v216 :: v_dual_mul_f32 v53, v35, v217
	v_dual_mul_f32 v62, v44, v216 :: v_dual_mul_f32 v63, v55, v217
	v_dual_fmac_f32 v32, v25, v209 :: v_dual_fmac_f32 v33, v34, v208
	v_dual_fmac_f32 v42, v45, v209 :: v_dual_fmac_f32 v43, v54, v208
	v_dual_fmac_f32 v52, v25, v217 :: v_dual_fmac_f32 v53, v34, v216
	v_dual_fmac_f32 v62, v45, v217 :: v_dual_fmac_f32 v63, v54, v216
	v_dual_fmac_f32 v32, v26, v210 :: v_dual_fmac_f32 v33, v36, v210
	v_dual_fmac_f32 v42, v46, v210 :: v_dual_fmac_f32 v43, v56, v210
	v_dual_fmac_f32 v52, v26, v218 :: v_dual_fmac_f32 v53, v36, v218
	v_dual_fmac_f32 v62, v46, v218 :: v_dual_fmac_f32 v63, v56, v218
	v_dual_fmac_f32 v32, v27, v211 :: v_dual_fmac_f32 v33, v37, v211
	v_dual_fmac_f32 v42, v47, v211 :: v_dual_fmac_f32 v43, v57, v211
	v_dual_fmac_f32 v52, v27, v219 :: v_dual_fmac_f32 v53, v37, v219
	v_dual_fmac_f32 v62, v47, v219 :: v_dual_fmac_f32 v63, v57, v219
	v_dual_fmac_f32 v32, v28, v212 :: v_dual_fmac_f32 v33, v38, v212
	v_dual_fmac_f32 v42, v48, v212 :: v_dual_fmac_f32 v43, v58, v212
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v220 :: v_dual_fmac_f32 v53, v38, v220
	v_dual_fmac_f32 v62, v48, v220 :: v_dual_fmac_f32 v63, v58, v220
	v_dual_fmac_f32 v32, v29, v213 :: v_dual_fmac_f32 v33, v39, v213
	v_dual_fmac_f32 v42, v49, v213 :: v_dual_fmac_f32 v43, v59, v213
	v_dual_fmac_f32 v52, v29, v221 :: v_dual_fmac_f32 v53, v39, v221
	v_dual_fmac_f32 v62, v49, v221 :: v_dual_fmac_f32 v63, v59, v221
	v_dual_fmac_f32 v32, v30, v214 :: v_dual_fmac_f32 v33, v40, v214
	v_dual_fmac_f32 v42, v50, v214 :: v_dual_fmac_f32 v43, v60, v214
	v_dual_fmac_f32 v52, v30, v222 :: v_dual_fmac_f32 v53, v40, v222
	v_dual_fmac_f32 v62, v50, v222 :: v_dual_fmac_f32 v63, v60, v222
	v_dual_fmac_f32 v32, v31, v215 :: v_dual_fmac_f32 v33, v41, v215
	v_dual_fmac_f32 v42, v51, v215 :: v_dual_fmac_f32 v43, v61, v215
	v_dual_fmac_f32 v52, v31, v223 :: v_dual_fmac_f32 v53, v41, v223
	v_dual_fmac_f32 v62, v51, v223 :: v_dual_fmac_f32 v63, v61, v223
	global_load_b128 v[208:211], v3, s[40:41]
	global_load_b128 v[212:215], v3, s[40:41] offset:16
	global_load_b128 v[216:219], v3, s[42:43]
	global_load_b128 v[220:223], v3, s[42:43] offset:16
	v_dual_add_f32 v136, v32, v136 :: v_dual_add_f32 v137, v137, v33
	v_dual_add_f32 v138, v42, v138 :: v_dual_add_f32 v139, v139, v43
	v_dual_add_f32 v148, v52, v148 :: v_dual_add_f32 v149, v149, v53
	v_dual_add_f32 v150, v62, v150 :: v_dual_add_f32 v151, v151, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s0
	s_wait_loadcnt 0x0
	.Lrx_b8_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[160:163], v2, s[28:29]
	global_load_b128 v[164:167], v2, s[28:29] offset:16
	global_load_b128 v[168:171], v2, s[30:31]
	global_load_b128 v[172:175], v2, s[30:31] offset:16
	global_load_b128 v[176:179], v2, s[32:33]
	global_load_b128 v[180:183], v2, s[32:33] offset:16
	global_load_b128 v[184:187], v2, s[34:35]
	global_load_b128 v[188:191], v2, s[34:35] offset:16
	global_load_b128 v[192:195], v2, s[36:37]
	global_load_b128 v[196:199], v2, s[36:37] offset:16
	global_load_b128 v[200:203], v2, s[38:39]
	global_load_b128 v[204:207], v2, s[38:39] offset:16
	global_load_b128 v[208:211], v2, s[40:41]
	global_load_b128 v[212:215], v2, s[40:41] offset:16
	global_load_b128 v[216:219], v2, s[42:43]
	global_load_b128 v[220:223], v2, s[42:43] offset:16
	.Lrx_b8_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x16
	v_bfe_u32 v24, v20, 0, 4
	v_bfe_u32 v25, v20, 4, 4
	v_bfe_u32 v26, v20, 8, 4
	v_bfe_u32 v27, v20, 12, 4
	v_bfe_u32 v28, v20, 16, 4
	v_bfe_u32 v29, v20, 20, 4
	v_bfe_u32 v30, v20, 24, 4
	v_bfe_u32 v31, v20, 28, 4
	v_cvt_f32_ubyte0_e32 v24, v24
	v_cvt_f32_ubyte0_e32 v25, v25
	v_cvt_f32_ubyte0_e32 v26, v26
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f32_ubyte0_e32 v28, v28
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f32_ubyte0_e32 v30, v30
	v_cvt_f32_ubyte0_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v34, v21, 0, 4
	v_bfe_u32 v35, v21, 4, 4
	v_bfe_u32 v36, v21, 8, 4
	v_bfe_u32 v37, v21, 12, 4
	v_bfe_u32 v38, v21, 16, 4
	v_bfe_u32 v39, v21, 20, 4
	v_bfe_u32 v40, v21, 24, 4
	v_bfe_u32 v41, v21, 28, 4
	v_cvt_f32_ubyte0_e32 v34, v34
	v_cvt_f32_ubyte0_e32 v35, v35
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v44, v22, 0, 4
	v_bfe_u32 v45, v22, 4, 4
	v_bfe_u32 v46, v22, 8, 4
	v_bfe_u32 v47, v22, 12, 4
	v_bfe_u32 v48, v22, 16, 4
	v_bfe_u32 v49, v22, 20, 4
	v_bfe_u32 v50, v22, 24, 4
	v_bfe_u32 v51, v22, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v54, v23, 0, 4
	v_bfe_u32 v55, v23, 4, 4
	v_bfe_u32 v56, v23, 8, 4
	v_bfe_u32 v57, v23, 12, 4
	v_bfe_u32 v58, v23, 16, 4
	v_bfe_u32 v59, v23, 20, 4
	v_bfe_u32 v60, v23, 24, 4
	v_bfe_u32 v61, v23, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v160 :: v_dual_mul_f32 v33, v35, v161
	v_dual_mul_f32 v42, v44, v160 :: v_dual_mul_f32 v43, v55, v161
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v168 :: v_dual_mul_f32 v53, v35, v169
	v_dual_mul_f32 v62, v44, v168 :: v_dual_mul_f32 v63, v55, v169
	v_dual_fmac_f32 v32, v25, v161 :: v_dual_fmac_f32 v33, v34, v160
	v_dual_fmac_f32 v42, v45, v161 :: v_dual_fmac_f32 v43, v54, v160
	v_dual_fmac_f32 v52, v25, v169 :: v_dual_fmac_f32 v53, v34, v168
	v_dual_fmac_f32 v62, v45, v169 :: v_dual_fmac_f32 v63, v54, v168
	v_dual_fmac_f32 v32, v26, v162 :: v_dual_fmac_f32 v33, v36, v162
	v_dual_fmac_f32 v42, v46, v162 :: v_dual_fmac_f32 v43, v56, v162
	v_dual_fmac_f32 v52, v26, v170 :: v_dual_fmac_f32 v53, v36, v170
	v_dual_fmac_f32 v62, v46, v170 :: v_dual_fmac_f32 v63, v56, v170
	v_dual_fmac_f32 v32, v27, v163 :: v_dual_fmac_f32 v33, v37, v163
	v_dual_fmac_f32 v42, v47, v163 :: v_dual_fmac_f32 v43, v57, v163
	v_dual_fmac_f32 v52, v27, v171 :: v_dual_fmac_f32 v53, v37, v171
	v_dual_fmac_f32 v62, v47, v171 :: v_dual_fmac_f32 v63, v57, v171
	v_dual_fmac_f32 v32, v28, v164 :: v_dual_fmac_f32 v33, v38, v164
	v_dual_fmac_f32 v42, v48, v164 :: v_dual_fmac_f32 v43, v58, v164
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v172 :: v_dual_fmac_f32 v53, v38, v172
	v_dual_fmac_f32 v62, v48, v172 :: v_dual_fmac_f32 v63, v58, v172
	v_dual_fmac_f32 v32, v29, v165 :: v_dual_fmac_f32 v33, v39, v165
	v_dual_fmac_f32 v42, v49, v165 :: v_dual_fmac_f32 v43, v59, v165
	v_dual_fmac_f32 v52, v29, v173 :: v_dual_fmac_f32 v53, v39, v173
	v_dual_fmac_f32 v62, v49, v173 :: v_dual_fmac_f32 v63, v59, v173
	v_dual_fmac_f32 v32, v30, v166 :: v_dual_fmac_f32 v33, v40, v166
	v_dual_fmac_f32 v42, v50, v166 :: v_dual_fmac_f32 v43, v60, v166
	v_dual_fmac_f32 v52, v30, v174 :: v_dual_fmac_f32 v53, v40, v174
	v_dual_fmac_f32 v62, v50, v174 :: v_dual_fmac_f32 v63, v60, v174
	v_dual_fmac_f32 v32, v31, v167 :: v_dual_fmac_f32 v33, v41, v167
	v_dual_fmac_f32 v42, v51, v167 :: v_dual_fmac_f32 v43, v61, v167
	v_dual_fmac_f32 v52, v31, v175 :: v_dual_fmac_f32 v53, v41, v175
	v_dual_fmac_f32 v62, v51, v175 :: v_dual_fmac_f32 v63, v61, v175
	global_load_b128 v[160:163], v3, s[28:29]
	global_load_b128 v[164:167], v3, s[28:29] offset:16
	global_load_b128 v[168:171], v3, s[30:31]
	global_load_b128 v[172:175], v3, s[30:31] offset:16
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v176 :: v_dual_mul_f32 v33, v35, v177
	v_dual_mul_f32 v42, v44, v176 :: v_dual_mul_f32 v43, v55, v177
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v184 :: v_dual_mul_f32 v53, v35, v185
	v_dual_mul_f32 v62, v44, v184 :: v_dual_mul_f32 v63, v55, v185
	v_dual_fmac_f32 v32, v25, v177 :: v_dual_fmac_f32 v33, v34, v176
	v_dual_fmac_f32 v42, v45, v177 :: v_dual_fmac_f32 v43, v54, v176
	v_dual_fmac_f32 v52, v25, v185 :: v_dual_fmac_f32 v53, v34, v184
	v_dual_fmac_f32 v62, v45, v185 :: v_dual_fmac_f32 v63, v54, v184
	v_dual_fmac_f32 v32, v26, v178 :: v_dual_fmac_f32 v33, v36, v178
	v_dual_fmac_f32 v42, v46, v178 :: v_dual_fmac_f32 v43, v56, v178
	v_dual_fmac_f32 v52, v26, v186 :: v_dual_fmac_f32 v53, v36, v186
	v_dual_fmac_f32 v62, v46, v186 :: v_dual_fmac_f32 v63, v56, v186
	v_dual_fmac_f32 v32, v27, v179 :: v_dual_fmac_f32 v33, v37, v179
	v_dual_fmac_f32 v42, v47, v179 :: v_dual_fmac_f32 v43, v57, v179
	v_dual_fmac_f32 v52, v27, v187 :: v_dual_fmac_f32 v53, v37, v187
	v_dual_fmac_f32 v62, v47, v187 :: v_dual_fmac_f32 v63, v57, v187
	v_dual_fmac_f32 v32, v28, v180 :: v_dual_fmac_f32 v33, v38, v180
	v_dual_fmac_f32 v42, v48, v180 :: v_dual_fmac_f32 v43, v58, v180
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v188 :: v_dual_fmac_f32 v53, v38, v188
	v_dual_fmac_f32 v62, v48, v188 :: v_dual_fmac_f32 v63, v58, v188
	v_dual_fmac_f32 v32, v29, v181 :: v_dual_fmac_f32 v33, v39, v181
	v_dual_fmac_f32 v42, v49, v181 :: v_dual_fmac_f32 v43, v59, v181
	v_dual_fmac_f32 v52, v29, v189 :: v_dual_fmac_f32 v53, v39, v189
	v_dual_fmac_f32 v62, v49, v189 :: v_dual_fmac_f32 v63, v59, v189
	v_dual_fmac_f32 v32, v30, v182 :: v_dual_fmac_f32 v33, v40, v182
	v_dual_fmac_f32 v42, v50, v182 :: v_dual_fmac_f32 v43, v60, v182
	v_dual_fmac_f32 v52, v30, v190 :: v_dual_fmac_f32 v53, v40, v190
	v_dual_fmac_f32 v62, v50, v190 :: v_dual_fmac_f32 v63, v60, v190
	v_dual_fmac_f32 v32, v31, v183 :: v_dual_fmac_f32 v33, v41, v183
	v_dual_fmac_f32 v42, v51, v183 :: v_dual_fmac_f32 v43, v61, v183
	v_dual_fmac_f32 v52, v31, v191 :: v_dual_fmac_f32 v53, v41, v191
	v_dual_fmac_f32 v62, v51, v191 :: v_dual_fmac_f32 v63, v61, v191
	global_load_b128 v[176:179], v3, s[32:33]
	global_load_b128 v[180:183], v3, s[32:33] offset:16
	global_load_b128 v[184:187], v3, s[34:35]
	global_load_b128 v[188:191], v3, s[34:35] offset:16
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	v_dual_add_f32 v104, v52, v104 :: v_dual_add_f32 v105, v105, v53
	v_dual_add_f32 v106, v62, v106 :: v_dual_add_f32 v107, v107, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v192 :: v_dual_mul_f32 v33, v35, v193
	v_dual_mul_f32 v42, v44, v192 :: v_dual_mul_f32 v43, v55, v193
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v200 :: v_dual_mul_f32 v53, v35, v201
	v_dual_mul_f32 v62, v44, v200 :: v_dual_mul_f32 v63, v55, v201
	v_dual_fmac_f32 v32, v25, v193 :: v_dual_fmac_f32 v33, v34, v192
	v_dual_fmac_f32 v42, v45, v193 :: v_dual_fmac_f32 v43, v54, v192
	v_dual_fmac_f32 v52, v25, v201 :: v_dual_fmac_f32 v53, v34, v200
	v_dual_fmac_f32 v62, v45, v201 :: v_dual_fmac_f32 v63, v54, v200
	v_dual_fmac_f32 v32, v26, v194 :: v_dual_fmac_f32 v33, v36, v194
	v_dual_fmac_f32 v42, v46, v194 :: v_dual_fmac_f32 v43, v56, v194
	v_dual_fmac_f32 v52, v26, v202 :: v_dual_fmac_f32 v53, v36, v202
	v_dual_fmac_f32 v62, v46, v202 :: v_dual_fmac_f32 v63, v56, v202
	v_dual_fmac_f32 v32, v27, v195 :: v_dual_fmac_f32 v33, v37, v195
	v_dual_fmac_f32 v42, v47, v195 :: v_dual_fmac_f32 v43, v57, v195
	v_dual_fmac_f32 v52, v27, v203 :: v_dual_fmac_f32 v53, v37, v203
	v_dual_fmac_f32 v62, v47, v203 :: v_dual_fmac_f32 v63, v57, v203
	v_dual_fmac_f32 v32, v28, v196 :: v_dual_fmac_f32 v33, v38, v196
	v_dual_fmac_f32 v42, v48, v196 :: v_dual_fmac_f32 v43, v58, v196
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v204 :: v_dual_fmac_f32 v53, v38, v204
	v_dual_fmac_f32 v62, v48, v204 :: v_dual_fmac_f32 v63, v58, v204
	v_dual_fmac_f32 v32, v29, v197 :: v_dual_fmac_f32 v33, v39, v197
	v_dual_fmac_f32 v42, v49, v197 :: v_dual_fmac_f32 v43, v59, v197
	v_dual_fmac_f32 v52, v29, v205 :: v_dual_fmac_f32 v53, v39, v205
	v_dual_fmac_f32 v62, v49, v205 :: v_dual_fmac_f32 v63, v59, v205
	v_dual_fmac_f32 v32, v30, v198 :: v_dual_fmac_f32 v33, v40, v198
	v_dual_fmac_f32 v42, v50, v198 :: v_dual_fmac_f32 v43, v60, v198
	v_dual_fmac_f32 v52, v30, v206 :: v_dual_fmac_f32 v53, v40, v206
	v_dual_fmac_f32 v62, v50, v206 :: v_dual_fmac_f32 v63, v60, v206
	v_dual_fmac_f32 v32, v31, v199 :: v_dual_fmac_f32 v33, v41, v199
	v_dual_fmac_f32 v42, v51, v199 :: v_dual_fmac_f32 v43, v61, v199
	v_dual_fmac_f32 v52, v31, v207 :: v_dual_fmac_f32 v53, v41, v207
	v_dual_fmac_f32 v62, v51, v207 :: v_dual_fmac_f32 v63, v61, v207
	global_load_b128 v[192:195], v3, s[36:37]
	global_load_b128 v[196:199], v3, s[36:37] offset:16
	global_load_b128 v[200:203], v3, s[38:39]
	global_load_b128 v[204:207], v3, s[38:39] offset:16
	v_dual_add_f32 v116, v32, v116 :: v_dual_add_f32 v117, v117, v33
	v_dual_add_f32 v118, v42, v118 :: v_dual_add_f32 v119, v119, v43
	v_dual_add_f32 v128, v52, v128 :: v_dual_add_f32 v129, v129, v53
	v_dual_add_f32 v130, v62, v130 :: v_dual_add_f32 v131, v131, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v208 :: v_dual_mul_f32 v33, v35, v209
	v_dual_mul_f32 v42, v44, v208 :: v_dual_mul_f32 v43, v55, v209
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v216 :: v_dual_mul_f32 v53, v35, v217
	v_dual_mul_f32 v62, v44, v216 :: v_dual_mul_f32 v63, v55, v217
	v_dual_fmac_f32 v32, v25, v209 :: v_dual_fmac_f32 v33, v34, v208
	v_dual_fmac_f32 v42, v45, v209 :: v_dual_fmac_f32 v43, v54, v208
	v_dual_fmac_f32 v52, v25, v217 :: v_dual_fmac_f32 v53, v34, v216
	v_dual_fmac_f32 v62, v45, v217 :: v_dual_fmac_f32 v63, v54, v216
	v_dual_fmac_f32 v32, v26, v210 :: v_dual_fmac_f32 v33, v36, v210
	v_dual_fmac_f32 v42, v46, v210 :: v_dual_fmac_f32 v43, v56, v210
	v_dual_fmac_f32 v52, v26, v218 :: v_dual_fmac_f32 v53, v36, v218
	v_dual_fmac_f32 v62, v46, v218 :: v_dual_fmac_f32 v63, v56, v218
	v_dual_fmac_f32 v32, v27, v211 :: v_dual_fmac_f32 v33, v37, v211
	v_dual_fmac_f32 v42, v47, v211 :: v_dual_fmac_f32 v43, v57, v211
	v_dual_fmac_f32 v52, v27, v219 :: v_dual_fmac_f32 v53, v37, v219
	v_dual_fmac_f32 v62, v47, v219 :: v_dual_fmac_f32 v63, v57, v219
	v_dual_fmac_f32 v32, v28, v212 :: v_dual_fmac_f32 v33, v38, v212
	v_dual_fmac_f32 v42, v48, v212 :: v_dual_fmac_f32 v43, v58, v212
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v220 :: v_dual_fmac_f32 v53, v38, v220
	v_dual_fmac_f32 v62, v48, v220 :: v_dual_fmac_f32 v63, v58, v220
	v_dual_fmac_f32 v32, v29, v213 :: v_dual_fmac_f32 v33, v39, v213
	v_dual_fmac_f32 v42, v49, v213 :: v_dual_fmac_f32 v43, v59, v213
	v_dual_fmac_f32 v52, v29, v221 :: v_dual_fmac_f32 v53, v39, v221
	v_dual_fmac_f32 v62, v49, v221 :: v_dual_fmac_f32 v63, v59, v221
	v_dual_fmac_f32 v32, v30, v214 :: v_dual_fmac_f32 v33, v40, v214
	v_dual_fmac_f32 v42, v50, v214 :: v_dual_fmac_f32 v43, v60, v214
	v_dual_fmac_f32 v52, v30, v222 :: v_dual_fmac_f32 v53, v40, v222
	v_dual_fmac_f32 v62, v50, v222 :: v_dual_fmac_f32 v63, v60, v222
	v_dual_fmac_f32 v32, v31, v215 :: v_dual_fmac_f32 v33, v41, v215
	v_dual_fmac_f32 v42, v51, v215 :: v_dual_fmac_f32 v43, v61, v215
	v_dual_fmac_f32 v52, v31, v223 :: v_dual_fmac_f32 v53, v41, v223
	v_dual_fmac_f32 v62, v51, v223 :: v_dual_fmac_f32 v63, v61, v223
	global_load_b128 v[208:211], v3, s[40:41]
	global_load_b128 v[212:215], v3, s[40:41] offset:16
	global_load_b128 v[216:219], v3, s[42:43]
	global_load_b128 v[220:223], v3, s[42:43] offset:16
	v_dual_add_f32 v140, v32, v140 :: v_dual_add_f32 v141, v141, v33
	v_dual_add_f32 v142, v42, v142 :: v_dual_add_f32 v143, v143, v43
	v_dual_add_f32 v152, v52, v152 :: v_dual_add_f32 v153, v153, v53
	v_dual_add_f32 v154, v62, v154 :: v_dual_add_f32 v155, v155, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s1
	s_wait_loadcnt 0x0
	.Lrx_b8_s1_done:
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_dual_add_f32 v100, v100, v104 :: v_dual_add_f32 v101, v101, v105
	v_dual_add_f32 v102, v102, v106 :: v_dual_add_f32 v103, v103, v107
	v_dual_add_f32 v112, v112, v116 :: v_dual_add_f32 v113, v113, v117
	v_dual_add_f32 v114, v114, v118 :: v_dual_add_f32 v115, v115, v119
	v_dual_add_f32 v124, v124, v128 :: v_dual_add_f32 v125, v125, v129
	v_dual_add_f32 v126, v126, v130 :: v_dual_add_f32 v127, v127, v131
	v_dual_add_f32 v136, v136, v140 :: v_dual_add_f32 v137, v137, v141
	v_dual_add_f32 v138, v138, v142 :: v_dual_add_f32 v139, v139, v143
	v_dual_add_f32 v148, v148, v152 :: v_dual_add_f32 v149, v149, v153
	v_dual_add_f32 v150, v150, v154 :: v_dual_add_f32 v151, v151, v155
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[160:163], v2, s[28:29]
	global_load_b128 v[164:167], v2, s[28:29] offset:16
	global_load_b128 v[168:171], v2, s[30:31]
	global_load_b128 v[172:175], v2, s[30:31] offset:16
	global_load_b128 v[176:179], v2, s[32:33]
	global_load_b128 v[180:183], v2, s[32:33] offset:16
	global_load_b128 v[184:187], v2, s[34:35]
	global_load_b128 v[188:191], v2, s[34:35] offset:16
	global_load_b128 v[192:195], v2, s[36:37]
	global_load_b128 v[196:199], v2, s[36:37] offset:16
	global_load_b128 v[200:203], v2, s[38:39]
	global_load_b128 v[204:207], v2, s[38:39] offset:16
	global_load_b128 v[208:211], v2, s[40:41]
	global_load_b128 v[212:215], v2, s[40:41] offset:16
	global_load_b128 v[216:219], v2, s[42:43]
	global_load_b128 v[220:223], v2, s[42:43] offset:16
	.Lrx_b8_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x16
	v_bfe_u32 v24, v20, 0, 4
	v_bfe_u32 v25, v20, 4, 4
	v_bfe_u32 v26, v20, 8, 4
	v_bfe_u32 v27, v20, 12, 4
	v_bfe_u32 v28, v20, 16, 4
	v_bfe_u32 v29, v20, 20, 4
	v_bfe_u32 v30, v20, 24, 4
	v_bfe_u32 v31, v20, 28, 4
	v_cvt_f32_ubyte0_e32 v24, v24
	v_cvt_f32_ubyte0_e32 v25, v25
	v_cvt_f32_ubyte0_e32 v26, v26
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f32_ubyte0_e32 v28, v28
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f32_ubyte0_e32 v30, v30
	v_cvt_f32_ubyte0_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v34, v21, 0, 4
	v_bfe_u32 v35, v21, 4, 4
	v_bfe_u32 v36, v21, 8, 4
	v_bfe_u32 v37, v21, 12, 4
	v_bfe_u32 v38, v21, 16, 4
	v_bfe_u32 v39, v21, 20, 4
	v_bfe_u32 v40, v21, 24, 4
	v_bfe_u32 v41, v21, 28, 4
	v_cvt_f32_ubyte0_e32 v34, v34
	v_cvt_f32_ubyte0_e32 v35, v35
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v44, v22, 0, 4
	v_bfe_u32 v45, v22, 4, 4
	v_bfe_u32 v46, v22, 8, 4
	v_bfe_u32 v47, v22, 12, 4
	v_bfe_u32 v48, v22, 16, 4
	v_bfe_u32 v49, v22, 20, 4
	v_bfe_u32 v50, v22, 24, 4
	v_bfe_u32 v51, v22, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v54, v23, 0, 4
	v_bfe_u32 v55, v23, 4, 4
	v_bfe_u32 v56, v23, 8, 4
	v_bfe_u32 v57, v23, 12, 4
	v_bfe_u32 v58, v23, 16, 4
	v_bfe_u32 v59, v23, 20, 4
	v_bfe_u32 v60, v23, 24, 4
	v_bfe_u32 v61, v23, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v160 :: v_dual_mul_f32 v33, v35, v161
	v_dual_mul_f32 v42, v44, v160 :: v_dual_mul_f32 v43, v55, v161
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v168 :: v_dual_mul_f32 v53, v35, v169
	v_dual_mul_f32 v62, v44, v168 :: v_dual_mul_f32 v63, v55, v169
	v_dual_fmac_f32 v32, v25, v161 :: v_dual_fmac_f32 v33, v34, v160
	v_dual_fmac_f32 v42, v45, v161 :: v_dual_fmac_f32 v43, v54, v160
	v_dual_fmac_f32 v52, v25, v169 :: v_dual_fmac_f32 v53, v34, v168
	v_dual_fmac_f32 v62, v45, v169 :: v_dual_fmac_f32 v63, v54, v168
	v_dual_fmac_f32 v32, v26, v162 :: v_dual_fmac_f32 v33, v36, v162
	v_dual_fmac_f32 v42, v46, v162 :: v_dual_fmac_f32 v43, v56, v162
	v_dual_fmac_f32 v52, v26, v170 :: v_dual_fmac_f32 v53, v36, v170
	v_dual_fmac_f32 v62, v46, v170 :: v_dual_fmac_f32 v63, v56, v170
	v_dual_fmac_f32 v32, v27, v163 :: v_dual_fmac_f32 v33, v37, v163
	v_dual_fmac_f32 v42, v47, v163 :: v_dual_fmac_f32 v43, v57, v163
	v_dual_fmac_f32 v52, v27, v171 :: v_dual_fmac_f32 v53, v37, v171
	v_dual_fmac_f32 v62, v47, v171 :: v_dual_fmac_f32 v63, v57, v171
	v_dual_fmac_f32 v32, v28, v164 :: v_dual_fmac_f32 v33, v38, v164
	v_dual_fmac_f32 v42, v48, v164 :: v_dual_fmac_f32 v43, v58, v164
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v172 :: v_dual_fmac_f32 v53, v38, v172
	v_dual_fmac_f32 v62, v48, v172 :: v_dual_fmac_f32 v63, v58, v172
	v_dual_fmac_f32 v32, v29, v165 :: v_dual_fmac_f32 v33, v39, v165
	v_dual_fmac_f32 v42, v49, v165 :: v_dual_fmac_f32 v43, v59, v165
	v_dual_fmac_f32 v52, v29, v173 :: v_dual_fmac_f32 v53, v39, v173
	v_dual_fmac_f32 v62, v49, v173 :: v_dual_fmac_f32 v63, v59, v173
	v_dual_fmac_f32 v32, v30, v166 :: v_dual_fmac_f32 v33, v40, v166
	v_dual_fmac_f32 v42, v50, v166 :: v_dual_fmac_f32 v43, v60, v166
	v_dual_fmac_f32 v52, v30, v174 :: v_dual_fmac_f32 v53, v40, v174
	v_dual_fmac_f32 v62, v50, v174 :: v_dual_fmac_f32 v63, v60, v174
	v_dual_fmac_f32 v32, v31, v167 :: v_dual_fmac_f32 v33, v41, v167
	v_dual_fmac_f32 v42, v51, v167 :: v_dual_fmac_f32 v43, v61, v167
	v_dual_fmac_f32 v52, v31, v175 :: v_dual_fmac_f32 v53, v41, v175
	v_dual_fmac_f32 v62, v51, v175 :: v_dual_fmac_f32 v63, v61, v175
	global_load_b128 v[160:163], v3, s[28:29]
	global_load_b128 v[164:167], v3, s[28:29] offset:16
	global_load_b128 v[168:171], v3, s[30:31]
	global_load_b128 v[172:175], v3, s[30:31] offset:16
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v176 :: v_dual_mul_f32 v33, v35, v177
	v_dual_mul_f32 v42, v44, v176 :: v_dual_mul_f32 v43, v55, v177
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v184 :: v_dual_mul_f32 v53, v35, v185
	v_dual_mul_f32 v62, v44, v184 :: v_dual_mul_f32 v63, v55, v185
	v_dual_fmac_f32 v32, v25, v177 :: v_dual_fmac_f32 v33, v34, v176
	v_dual_fmac_f32 v42, v45, v177 :: v_dual_fmac_f32 v43, v54, v176
	v_dual_fmac_f32 v52, v25, v185 :: v_dual_fmac_f32 v53, v34, v184
	v_dual_fmac_f32 v62, v45, v185 :: v_dual_fmac_f32 v63, v54, v184
	v_dual_fmac_f32 v32, v26, v178 :: v_dual_fmac_f32 v33, v36, v178
	v_dual_fmac_f32 v42, v46, v178 :: v_dual_fmac_f32 v43, v56, v178
	v_dual_fmac_f32 v52, v26, v186 :: v_dual_fmac_f32 v53, v36, v186
	v_dual_fmac_f32 v62, v46, v186 :: v_dual_fmac_f32 v63, v56, v186
	v_dual_fmac_f32 v32, v27, v179 :: v_dual_fmac_f32 v33, v37, v179
	v_dual_fmac_f32 v42, v47, v179 :: v_dual_fmac_f32 v43, v57, v179
	v_dual_fmac_f32 v52, v27, v187 :: v_dual_fmac_f32 v53, v37, v187
	v_dual_fmac_f32 v62, v47, v187 :: v_dual_fmac_f32 v63, v57, v187
	v_dual_fmac_f32 v32, v28, v180 :: v_dual_fmac_f32 v33, v38, v180
	v_dual_fmac_f32 v42, v48, v180 :: v_dual_fmac_f32 v43, v58, v180
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v188 :: v_dual_fmac_f32 v53, v38, v188
	v_dual_fmac_f32 v62, v48, v188 :: v_dual_fmac_f32 v63, v58, v188
	v_dual_fmac_f32 v32, v29, v181 :: v_dual_fmac_f32 v33, v39, v181
	v_dual_fmac_f32 v42, v49, v181 :: v_dual_fmac_f32 v43, v59, v181
	v_dual_fmac_f32 v52, v29, v189 :: v_dual_fmac_f32 v53, v39, v189
	v_dual_fmac_f32 v62, v49, v189 :: v_dual_fmac_f32 v63, v59, v189
	v_dual_fmac_f32 v32, v30, v182 :: v_dual_fmac_f32 v33, v40, v182
	v_dual_fmac_f32 v42, v50, v182 :: v_dual_fmac_f32 v43, v60, v182
	v_dual_fmac_f32 v52, v30, v190 :: v_dual_fmac_f32 v53, v40, v190
	v_dual_fmac_f32 v62, v50, v190 :: v_dual_fmac_f32 v63, v60, v190
	v_dual_fmac_f32 v32, v31, v183 :: v_dual_fmac_f32 v33, v41, v183
	v_dual_fmac_f32 v42, v51, v183 :: v_dual_fmac_f32 v43, v61, v183
	v_dual_fmac_f32 v52, v31, v191 :: v_dual_fmac_f32 v53, v41, v191
	v_dual_fmac_f32 v62, v51, v191 :: v_dual_fmac_f32 v63, v61, v191
	global_load_b128 v[176:179], v3, s[32:33]
	global_load_b128 v[180:183], v3, s[32:33] offset:16
	global_load_b128 v[184:187], v3, s[34:35]
	global_load_b128 v[188:191], v3, s[34:35] offset:16
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	v_dual_add_f32 v104, v52, v104 :: v_dual_add_f32 v105, v105, v53
	v_dual_add_f32 v106, v62, v106 :: v_dual_add_f32 v107, v107, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v192 :: v_dual_mul_f32 v33, v35, v193
	v_dual_mul_f32 v42, v44, v192 :: v_dual_mul_f32 v43, v55, v193
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v200 :: v_dual_mul_f32 v53, v35, v201
	v_dual_mul_f32 v62, v44, v200 :: v_dual_mul_f32 v63, v55, v201
	v_dual_fmac_f32 v32, v25, v193 :: v_dual_fmac_f32 v33, v34, v192
	v_dual_fmac_f32 v42, v45, v193 :: v_dual_fmac_f32 v43, v54, v192
	v_dual_fmac_f32 v52, v25, v201 :: v_dual_fmac_f32 v53, v34, v200
	v_dual_fmac_f32 v62, v45, v201 :: v_dual_fmac_f32 v63, v54, v200
	v_dual_fmac_f32 v32, v26, v194 :: v_dual_fmac_f32 v33, v36, v194
	v_dual_fmac_f32 v42, v46, v194 :: v_dual_fmac_f32 v43, v56, v194
	v_dual_fmac_f32 v52, v26, v202 :: v_dual_fmac_f32 v53, v36, v202
	v_dual_fmac_f32 v62, v46, v202 :: v_dual_fmac_f32 v63, v56, v202
	v_dual_fmac_f32 v32, v27, v195 :: v_dual_fmac_f32 v33, v37, v195
	v_dual_fmac_f32 v42, v47, v195 :: v_dual_fmac_f32 v43, v57, v195
	v_dual_fmac_f32 v52, v27, v203 :: v_dual_fmac_f32 v53, v37, v203
	v_dual_fmac_f32 v62, v47, v203 :: v_dual_fmac_f32 v63, v57, v203
	v_dual_fmac_f32 v32, v28, v196 :: v_dual_fmac_f32 v33, v38, v196
	v_dual_fmac_f32 v42, v48, v196 :: v_dual_fmac_f32 v43, v58, v196
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v204 :: v_dual_fmac_f32 v53, v38, v204
	v_dual_fmac_f32 v62, v48, v204 :: v_dual_fmac_f32 v63, v58, v204
	v_dual_fmac_f32 v32, v29, v197 :: v_dual_fmac_f32 v33, v39, v197
	v_dual_fmac_f32 v42, v49, v197 :: v_dual_fmac_f32 v43, v59, v197
	v_dual_fmac_f32 v52, v29, v205 :: v_dual_fmac_f32 v53, v39, v205
	v_dual_fmac_f32 v62, v49, v205 :: v_dual_fmac_f32 v63, v59, v205
	v_dual_fmac_f32 v32, v30, v198 :: v_dual_fmac_f32 v33, v40, v198
	v_dual_fmac_f32 v42, v50, v198 :: v_dual_fmac_f32 v43, v60, v198
	v_dual_fmac_f32 v52, v30, v206 :: v_dual_fmac_f32 v53, v40, v206
	v_dual_fmac_f32 v62, v50, v206 :: v_dual_fmac_f32 v63, v60, v206
	v_dual_fmac_f32 v32, v31, v199 :: v_dual_fmac_f32 v33, v41, v199
	v_dual_fmac_f32 v42, v51, v199 :: v_dual_fmac_f32 v43, v61, v199
	v_dual_fmac_f32 v52, v31, v207 :: v_dual_fmac_f32 v53, v41, v207
	v_dual_fmac_f32 v62, v51, v207 :: v_dual_fmac_f32 v63, v61, v207
	global_load_b128 v[192:195], v3, s[36:37]
	global_load_b128 v[196:199], v3, s[36:37] offset:16
	global_load_b128 v[200:203], v3, s[38:39]
	global_load_b128 v[204:207], v3, s[38:39] offset:16
	v_dual_add_f32 v116, v32, v116 :: v_dual_add_f32 v117, v117, v33
	v_dual_add_f32 v118, v42, v118 :: v_dual_add_f32 v119, v119, v43
	v_dual_add_f32 v128, v52, v128 :: v_dual_add_f32 v129, v129, v53
	v_dual_add_f32 v130, v62, v130 :: v_dual_add_f32 v131, v131, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v208 :: v_dual_mul_f32 v33, v35, v209
	v_dual_mul_f32 v42, v44, v208 :: v_dual_mul_f32 v43, v55, v209
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v216 :: v_dual_mul_f32 v53, v35, v217
	v_dual_mul_f32 v62, v44, v216 :: v_dual_mul_f32 v63, v55, v217
	v_dual_fmac_f32 v32, v25, v209 :: v_dual_fmac_f32 v33, v34, v208
	v_dual_fmac_f32 v42, v45, v209 :: v_dual_fmac_f32 v43, v54, v208
	v_dual_fmac_f32 v52, v25, v217 :: v_dual_fmac_f32 v53, v34, v216
	v_dual_fmac_f32 v62, v45, v217 :: v_dual_fmac_f32 v63, v54, v216
	v_dual_fmac_f32 v32, v26, v210 :: v_dual_fmac_f32 v33, v36, v210
	v_dual_fmac_f32 v42, v46, v210 :: v_dual_fmac_f32 v43, v56, v210
	v_dual_fmac_f32 v52, v26, v218 :: v_dual_fmac_f32 v53, v36, v218
	v_dual_fmac_f32 v62, v46, v218 :: v_dual_fmac_f32 v63, v56, v218
	v_dual_fmac_f32 v32, v27, v211 :: v_dual_fmac_f32 v33, v37, v211
	v_dual_fmac_f32 v42, v47, v211 :: v_dual_fmac_f32 v43, v57, v211
	v_dual_fmac_f32 v52, v27, v219 :: v_dual_fmac_f32 v53, v37, v219
	v_dual_fmac_f32 v62, v47, v219 :: v_dual_fmac_f32 v63, v57, v219
	v_dual_fmac_f32 v32, v28, v212 :: v_dual_fmac_f32 v33, v38, v212
	v_dual_fmac_f32 v42, v48, v212 :: v_dual_fmac_f32 v43, v58, v212
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v220 :: v_dual_fmac_f32 v53, v38, v220
	v_dual_fmac_f32 v62, v48, v220 :: v_dual_fmac_f32 v63, v58, v220
	v_dual_fmac_f32 v32, v29, v213 :: v_dual_fmac_f32 v33, v39, v213
	v_dual_fmac_f32 v42, v49, v213 :: v_dual_fmac_f32 v43, v59, v213
	v_dual_fmac_f32 v52, v29, v221 :: v_dual_fmac_f32 v53, v39, v221
	v_dual_fmac_f32 v62, v49, v221 :: v_dual_fmac_f32 v63, v59, v221
	v_dual_fmac_f32 v32, v30, v214 :: v_dual_fmac_f32 v33, v40, v214
	v_dual_fmac_f32 v42, v50, v214 :: v_dual_fmac_f32 v43, v60, v214
	v_dual_fmac_f32 v52, v30, v222 :: v_dual_fmac_f32 v53, v40, v222
	v_dual_fmac_f32 v62, v50, v222 :: v_dual_fmac_f32 v63, v60, v222
	v_dual_fmac_f32 v32, v31, v215 :: v_dual_fmac_f32 v33, v41, v215
	v_dual_fmac_f32 v42, v51, v215 :: v_dual_fmac_f32 v43, v61, v215
	v_dual_fmac_f32 v52, v31, v223 :: v_dual_fmac_f32 v53, v41, v223
	v_dual_fmac_f32 v62, v51, v223 :: v_dual_fmac_f32 v63, v61, v223
	global_load_b128 v[208:211], v3, s[40:41]
	global_load_b128 v[212:215], v3, s[40:41] offset:16
	global_load_b128 v[216:219], v3, s[42:43]
	global_load_b128 v[220:223], v3, s[42:43] offset:16
	v_dual_add_f32 v140, v32, v140 :: v_dual_add_f32 v141, v141, v33
	v_dual_add_f32 v142, v42, v142 :: v_dual_add_f32 v143, v143, v43
	v_dual_add_f32 v152, v52, v152 :: v_dual_add_f32 v153, v153, v53
	v_dual_add_f32 v154, v62, v154 :: v_dual_add_f32 v155, v155, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s2
	s_wait_loadcnt 0x0
	.Lrx_b8_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[160:163], v2, s[28:29]
	global_load_b128 v[164:167], v2, s[28:29] offset:16
	global_load_b128 v[168:171], v2, s[30:31]
	global_load_b128 v[172:175], v2, s[30:31] offset:16
	global_load_b128 v[176:179], v2, s[32:33]
	global_load_b128 v[180:183], v2, s[32:33] offset:16
	global_load_b128 v[184:187], v2, s[34:35]
	global_load_b128 v[188:191], v2, s[34:35] offset:16
	global_load_b128 v[192:195], v2, s[36:37]
	global_load_b128 v[196:199], v2, s[36:37] offset:16
	global_load_b128 v[200:203], v2, s[38:39]
	global_load_b128 v[204:207], v2, s[38:39] offset:16
	global_load_b128 v[208:211], v2, s[40:41]
	global_load_b128 v[212:215], v2, s[40:41] offset:16
	global_load_b128 v[216:219], v2, s[42:43]
	global_load_b128 v[220:223], v2, s[42:43] offset:16
	.Lrx_b8_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x16
	v_bfe_u32 v24, v20, 0, 4
	v_bfe_u32 v25, v20, 4, 4
	v_bfe_u32 v26, v20, 8, 4
	v_bfe_u32 v27, v20, 12, 4
	v_bfe_u32 v28, v20, 16, 4
	v_bfe_u32 v29, v20, 20, 4
	v_bfe_u32 v30, v20, 24, 4
	v_bfe_u32 v31, v20, 28, 4
	v_cvt_f32_ubyte0_e32 v24, v24
	v_cvt_f32_ubyte0_e32 v25, v25
	v_cvt_f32_ubyte0_e32 v26, v26
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f32_ubyte0_e32 v28, v28
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f32_ubyte0_e32 v30, v30
	v_cvt_f32_ubyte0_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x14
	v_bfe_u32 v34, v21, 0, 4
	v_bfe_u32 v35, v21, 4, 4
	v_bfe_u32 v36, v21, 8, 4
	v_bfe_u32 v37, v21, 12, 4
	v_bfe_u32 v38, v21, 16, 4
	v_bfe_u32 v39, v21, 20, 4
	v_bfe_u32 v40, v21, 24, 4
	v_bfe_u32 v41, v21, 28, 4
	v_cvt_f32_ubyte0_e32 v34, v34
	v_cvt_f32_ubyte0_e32 v35, v35
	v_cvt_f32_ubyte0_e32 v36, v36
	v_cvt_f32_ubyte0_e32 v37, v37
	v_cvt_f32_ubyte0_e32 v38, v38
	v_cvt_f32_ubyte0_e32 v39, v39
	v_cvt_f32_ubyte0_e32 v40, v40
	v_cvt_f32_ubyte0_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x12
	v_bfe_u32 v44, v22, 0, 4
	v_bfe_u32 v45, v22, 4, 4
	v_bfe_u32 v46, v22, 8, 4
	v_bfe_u32 v47, v22, 12, 4
	v_bfe_u32 v48, v22, 16, 4
	v_bfe_u32 v49, v22, 20, 4
	v_bfe_u32 v50, v22, 24, 4
	v_bfe_u32 v51, v22, 28, 4
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v46, v46
	v_cvt_f32_ubyte0_e32 v47, v47
	v_cvt_f32_ubyte0_e32 v48, v48
	v_cvt_f32_ubyte0_e32 v49, v49
	v_cvt_f32_ubyte0_e32 v50, v50
	v_cvt_f32_ubyte0_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x10
	v_bfe_u32 v54, v23, 0, 4
	v_bfe_u32 v55, v23, 4, 4
	v_bfe_u32 v56, v23, 8, 4
	v_bfe_u32 v57, v23, 12, 4
	v_bfe_u32 v58, v23, 16, 4
	v_bfe_u32 v59, v23, 20, 4
	v_bfe_u32 v60, v23, 24, 4
	v_bfe_u32 v61, v23, 28, 4
	v_cvt_f32_ubyte0_e32 v54, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v58, v58
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v160 :: v_dual_mul_f32 v33, v35, v161
	v_dual_mul_f32 v42, v44, v160 :: v_dual_mul_f32 v43, v55, v161
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v168 :: v_dual_mul_f32 v53, v35, v169
	v_dual_mul_f32 v62, v44, v168 :: v_dual_mul_f32 v63, v55, v169
	v_dual_fmac_f32 v32, v25, v161 :: v_dual_fmac_f32 v33, v34, v160
	v_dual_fmac_f32 v42, v45, v161 :: v_dual_fmac_f32 v43, v54, v160
	v_dual_fmac_f32 v52, v25, v169 :: v_dual_fmac_f32 v53, v34, v168
	v_dual_fmac_f32 v62, v45, v169 :: v_dual_fmac_f32 v63, v54, v168
	v_dual_fmac_f32 v32, v26, v162 :: v_dual_fmac_f32 v33, v36, v162
	v_dual_fmac_f32 v42, v46, v162 :: v_dual_fmac_f32 v43, v56, v162
	v_dual_fmac_f32 v52, v26, v170 :: v_dual_fmac_f32 v53, v36, v170
	v_dual_fmac_f32 v62, v46, v170 :: v_dual_fmac_f32 v63, v56, v170
	v_dual_fmac_f32 v32, v27, v163 :: v_dual_fmac_f32 v33, v37, v163
	v_dual_fmac_f32 v42, v47, v163 :: v_dual_fmac_f32 v43, v57, v163
	v_dual_fmac_f32 v52, v27, v171 :: v_dual_fmac_f32 v53, v37, v171
	v_dual_fmac_f32 v62, v47, v171 :: v_dual_fmac_f32 v63, v57, v171
	v_dual_fmac_f32 v32, v28, v164 :: v_dual_fmac_f32 v33, v38, v164
	v_dual_fmac_f32 v42, v48, v164 :: v_dual_fmac_f32 v43, v58, v164
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v172 :: v_dual_fmac_f32 v53, v38, v172
	v_dual_fmac_f32 v62, v48, v172 :: v_dual_fmac_f32 v63, v58, v172
	v_dual_fmac_f32 v32, v29, v165 :: v_dual_fmac_f32 v33, v39, v165
	v_dual_fmac_f32 v42, v49, v165 :: v_dual_fmac_f32 v43, v59, v165
	v_dual_fmac_f32 v52, v29, v173 :: v_dual_fmac_f32 v53, v39, v173
	v_dual_fmac_f32 v62, v49, v173 :: v_dual_fmac_f32 v63, v59, v173
	v_dual_fmac_f32 v32, v30, v166 :: v_dual_fmac_f32 v33, v40, v166
	v_dual_fmac_f32 v42, v50, v166 :: v_dual_fmac_f32 v43, v60, v166
	v_dual_fmac_f32 v52, v30, v174 :: v_dual_fmac_f32 v53, v40, v174
	v_dual_fmac_f32 v62, v50, v174 :: v_dual_fmac_f32 v63, v60, v174
	v_dual_fmac_f32 v32, v31, v167 :: v_dual_fmac_f32 v33, v41, v167
	v_dual_fmac_f32 v42, v51, v167 :: v_dual_fmac_f32 v43, v61, v167
	v_dual_fmac_f32 v52, v31, v175 :: v_dual_fmac_f32 v53, v41, v175
	v_dual_fmac_f32 v62, v51, v175 :: v_dual_fmac_f32 v63, v61, v175
	global_load_b128 v[160:163], v3, s[28:29]
	global_load_b128 v[164:167], v3, s[28:29] offset:16
	global_load_b128 v[168:171], v3, s[30:31]
	global_load_b128 v[172:175], v3, s[30:31] offset:16
	v_dual_add_f32 v72, v32, v72 :: v_dual_add_f32 v73, v73, v33
	v_dual_add_f32 v74, v42, v74 :: v_dual_add_f32 v75, v75, v43
	v_dual_add_f32 v84, v52, v84 :: v_dual_add_f32 v85, v85, v53
	v_dual_add_f32 v86, v62, v86 :: v_dual_add_f32 v87, v87, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v176 :: v_dual_mul_f32 v33, v35, v177
	v_dual_mul_f32 v42, v44, v176 :: v_dual_mul_f32 v43, v55, v177
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v184 :: v_dual_mul_f32 v53, v35, v185
	v_dual_mul_f32 v62, v44, v184 :: v_dual_mul_f32 v63, v55, v185
	v_dual_fmac_f32 v32, v25, v177 :: v_dual_fmac_f32 v33, v34, v176
	v_dual_fmac_f32 v42, v45, v177 :: v_dual_fmac_f32 v43, v54, v176
	v_dual_fmac_f32 v52, v25, v185 :: v_dual_fmac_f32 v53, v34, v184
	v_dual_fmac_f32 v62, v45, v185 :: v_dual_fmac_f32 v63, v54, v184
	v_dual_fmac_f32 v32, v26, v178 :: v_dual_fmac_f32 v33, v36, v178
	v_dual_fmac_f32 v42, v46, v178 :: v_dual_fmac_f32 v43, v56, v178
	v_dual_fmac_f32 v52, v26, v186 :: v_dual_fmac_f32 v53, v36, v186
	v_dual_fmac_f32 v62, v46, v186 :: v_dual_fmac_f32 v63, v56, v186
	v_dual_fmac_f32 v32, v27, v179 :: v_dual_fmac_f32 v33, v37, v179
	v_dual_fmac_f32 v42, v47, v179 :: v_dual_fmac_f32 v43, v57, v179
	v_dual_fmac_f32 v52, v27, v187 :: v_dual_fmac_f32 v53, v37, v187
	v_dual_fmac_f32 v62, v47, v187 :: v_dual_fmac_f32 v63, v57, v187
	v_dual_fmac_f32 v32, v28, v180 :: v_dual_fmac_f32 v33, v38, v180
	v_dual_fmac_f32 v42, v48, v180 :: v_dual_fmac_f32 v43, v58, v180
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v188 :: v_dual_fmac_f32 v53, v38, v188
	v_dual_fmac_f32 v62, v48, v188 :: v_dual_fmac_f32 v63, v58, v188
	v_dual_fmac_f32 v32, v29, v181 :: v_dual_fmac_f32 v33, v39, v181
	v_dual_fmac_f32 v42, v49, v181 :: v_dual_fmac_f32 v43, v59, v181
	v_dual_fmac_f32 v52, v29, v189 :: v_dual_fmac_f32 v53, v39, v189
	v_dual_fmac_f32 v62, v49, v189 :: v_dual_fmac_f32 v63, v59, v189
	v_dual_fmac_f32 v32, v30, v182 :: v_dual_fmac_f32 v33, v40, v182
	v_dual_fmac_f32 v42, v50, v182 :: v_dual_fmac_f32 v43, v60, v182
	v_dual_fmac_f32 v52, v30, v190 :: v_dual_fmac_f32 v53, v40, v190
	v_dual_fmac_f32 v62, v50, v190 :: v_dual_fmac_f32 v63, v60, v190
	v_dual_fmac_f32 v32, v31, v183 :: v_dual_fmac_f32 v33, v41, v183
	v_dual_fmac_f32 v42, v51, v183 :: v_dual_fmac_f32 v43, v61, v183
	v_dual_fmac_f32 v52, v31, v191 :: v_dual_fmac_f32 v53, v41, v191
	v_dual_fmac_f32 v62, v51, v191 :: v_dual_fmac_f32 v63, v61, v191
	global_load_b128 v[176:179], v3, s[32:33]
	global_load_b128 v[180:183], v3, s[32:33] offset:16
	global_load_b128 v[184:187], v3, s[34:35]
	global_load_b128 v[188:191], v3, s[34:35] offset:16
	v_dual_add_f32 v96, v32, v96 :: v_dual_add_f32 v97, v97, v33
	v_dual_add_f32 v98, v42, v98 :: v_dual_add_f32 v99, v99, v43
	v_dual_add_f32 v108, v52, v108 :: v_dual_add_f32 v109, v109, v53
	v_dual_add_f32 v110, v62, v110 :: v_dual_add_f32 v111, v111, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v192 :: v_dual_mul_f32 v33, v35, v193
	v_dual_mul_f32 v42, v44, v192 :: v_dual_mul_f32 v43, v55, v193
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v200 :: v_dual_mul_f32 v53, v35, v201
	v_dual_mul_f32 v62, v44, v200 :: v_dual_mul_f32 v63, v55, v201
	v_dual_fmac_f32 v32, v25, v193 :: v_dual_fmac_f32 v33, v34, v192
	v_dual_fmac_f32 v42, v45, v193 :: v_dual_fmac_f32 v43, v54, v192
	v_dual_fmac_f32 v52, v25, v201 :: v_dual_fmac_f32 v53, v34, v200
	v_dual_fmac_f32 v62, v45, v201 :: v_dual_fmac_f32 v63, v54, v200
	v_dual_fmac_f32 v32, v26, v194 :: v_dual_fmac_f32 v33, v36, v194
	v_dual_fmac_f32 v42, v46, v194 :: v_dual_fmac_f32 v43, v56, v194
	v_dual_fmac_f32 v52, v26, v202 :: v_dual_fmac_f32 v53, v36, v202
	v_dual_fmac_f32 v62, v46, v202 :: v_dual_fmac_f32 v63, v56, v202
	v_dual_fmac_f32 v32, v27, v195 :: v_dual_fmac_f32 v33, v37, v195
	v_dual_fmac_f32 v42, v47, v195 :: v_dual_fmac_f32 v43, v57, v195
	v_dual_fmac_f32 v52, v27, v203 :: v_dual_fmac_f32 v53, v37, v203
	v_dual_fmac_f32 v62, v47, v203 :: v_dual_fmac_f32 v63, v57, v203
	v_dual_fmac_f32 v32, v28, v196 :: v_dual_fmac_f32 v33, v38, v196
	v_dual_fmac_f32 v42, v48, v196 :: v_dual_fmac_f32 v43, v58, v196
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v204 :: v_dual_fmac_f32 v53, v38, v204
	v_dual_fmac_f32 v62, v48, v204 :: v_dual_fmac_f32 v63, v58, v204
	v_dual_fmac_f32 v32, v29, v197 :: v_dual_fmac_f32 v33, v39, v197
	v_dual_fmac_f32 v42, v49, v197 :: v_dual_fmac_f32 v43, v59, v197
	v_dual_fmac_f32 v52, v29, v205 :: v_dual_fmac_f32 v53, v39, v205
	v_dual_fmac_f32 v62, v49, v205 :: v_dual_fmac_f32 v63, v59, v205
	v_dual_fmac_f32 v32, v30, v198 :: v_dual_fmac_f32 v33, v40, v198
	v_dual_fmac_f32 v42, v50, v198 :: v_dual_fmac_f32 v43, v60, v198
	v_dual_fmac_f32 v52, v30, v206 :: v_dual_fmac_f32 v53, v40, v206
	v_dual_fmac_f32 v62, v50, v206 :: v_dual_fmac_f32 v63, v60, v206
	v_dual_fmac_f32 v32, v31, v199 :: v_dual_fmac_f32 v33, v41, v199
	v_dual_fmac_f32 v42, v51, v199 :: v_dual_fmac_f32 v43, v61, v199
	v_dual_fmac_f32 v52, v31, v207 :: v_dual_fmac_f32 v53, v41, v207
	v_dual_fmac_f32 v62, v51, v207 :: v_dual_fmac_f32 v63, v61, v207
	global_load_b128 v[192:195], v3, s[36:37]
	global_load_b128 v[196:199], v3, s[36:37] offset:16
	global_load_b128 v[200:203], v3, s[38:39]
	global_load_b128 v[204:207], v3, s[38:39] offset:16
	v_dual_add_f32 v120, v32, v120 :: v_dual_add_f32 v121, v121, v33
	v_dual_add_f32 v122, v42, v122 :: v_dual_add_f32 v123, v123, v43
	v_dual_add_f32 v132, v52, v132 :: v_dual_add_f32 v133, v133, v53
	v_dual_add_f32 v134, v62, v134 :: v_dual_add_f32 v135, v135, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v208 :: v_dual_mul_f32 v33, v35, v209
	v_dual_mul_f32 v42, v44, v208 :: v_dual_mul_f32 v43, v55, v209
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v216 :: v_dual_mul_f32 v53, v35, v217
	v_dual_mul_f32 v62, v44, v216 :: v_dual_mul_f32 v63, v55, v217
	v_dual_fmac_f32 v32, v25, v209 :: v_dual_fmac_f32 v33, v34, v208
	v_dual_fmac_f32 v42, v45, v209 :: v_dual_fmac_f32 v43, v54, v208
	v_dual_fmac_f32 v52, v25, v217 :: v_dual_fmac_f32 v53, v34, v216
	v_dual_fmac_f32 v62, v45, v217 :: v_dual_fmac_f32 v63, v54, v216
	v_dual_fmac_f32 v32, v26, v210 :: v_dual_fmac_f32 v33, v36, v210
	v_dual_fmac_f32 v42, v46, v210 :: v_dual_fmac_f32 v43, v56, v210
	v_dual_fmac_f32 v52, v26, v218 :: v_dual_fmac_f32 v53, v36, v218
	v_dual_fmac_f32 v62, v46, v218 :: v_dual_fmac_f32 v63, v56, v218
	v_dual_fmac_f32 v32, v27, v211 :: v_dual_fmac_f32 v33, v37, v211
	v_dual_fmac_f32 v42, v47, v211 :: v_dual_fmac_f32 v43, v57, v211
	v_dual_fmac_f32 v52, v27, v219 :: v_dual_fmac_f32 v53, v37, v219
	v_dual_fmac_f32 v62, v47, v219 :: v_dual_fmac_f32 v63, v57, v219
	v_dual_fmac_f32 v32, v28, v212 :: v_dual_fmac_f32 v33, v38, v212
	v_dual_fmac_f32 v42, v48, v212 :: v_dual_fmac_f32 v43, v58, v212
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v52, v28, v220 :: v_dual_fmac_f32 v53, v38, v220
	v_dual_fmac_f32 v62, v48, v220 :: v_dual_fmac_f32 v63, v58, v220
	v_dual_fmac_f32 v32, v29, v213 :: v_dual_fmac_f32 v33, v39, v213
	v_dual_fmac_f32 v42, v49, v213 :: v_dual_fmac_f32 v43, v59, v213
	v_dual_fmac_f32 v52, v29, v221 :: v_dual_fmac_f32 v53, v39, v221
	v_dual_fmac_f32 v62, v49, v221 :: v_dual_fmac_f32 v63, v59, v221
	v_dual_fmac_f32 v32, v30, v214 :: v_dual_fmac_f32 v33, v40, v214
	v_dual_fmac_f32 v42, v50, v214 :: v_dual_fmac_f32 v43, v60, v214
	v_dual_fmac_f32 v52, v30, v222 :: v_dual_fmac_f32 v53, v40, v222
	v_dual_fmac_f32 v62, v50, v222 :: v_dual_fmac_f32 v63, v60, v222
	v_dual_fmac_f32 v32, v31, v215 :: v_dual_fmac_f32 v33, v41, v215
	v_dual_fmac_f32 v42, v51, v215 :: v_dual_fmac_f32 v43, v61, v215
	v_dual_fmac_f32 v52, v31, v223 :: v_dual_fmac_f32 v53, v41, v223
	v_dual_fmac_f32 v62, v51, v223 :: v_dual_fmac_f32 v63, v61, v223
	global_load_b128 v[208:211], v3, s[40:41]
	global_load_b128 v[212:215], v3, s[40:41] offset:16
	global_load_b128 v[216:219], v3, s[42:43]
	global_load_b128 v[220:223], v3, s[42:43] offset:16
	v_dual_add_f32 v144, v32, v144 :: v_dual_add_f32 v145, v145, v33
	v_dual_add_f32 v146, v42, v146 :: v_dual_add_f32 v147, v147, v43
	v_dual_add_f32 v156, v52, v156 :: v_dual_add_f32 v157, v157, v53
	v_dual_add_f32 v158, v62, v158 :: v_dual_add_f32 v159, v159, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s3
	s_wait_loadcnt 0x0
	.Lrx_b8_s3_done:
	v_dual_add_f32 v68, v68, v72 :: v_dual_add_f32 v69, v69, v73
	v_dual_add_f32 v70, v70, v74 :: v_dual_add_f32 v71, v71, v75
	v_dual_add_f32 v80, v80, v84 :: v_dual_add_f32 v81, v81, v85
	v_dual_add_f32 v82, v82, v86 :: v_dual_add_f32 v83, v83, v87
	v_dual_add_f32 v92, v92, v96 :: v_dual_add_f32 v93, v93, v97
	v_dual_add_f32 v94, v94, v98 :: v_dual_add_f32 v95, v95, v99
	v_dual_add_f32 v104, v104, v108 :: v_dual_add_f32 v105, v105, v109
	v_dual_add_f32 v106, v106, v110 :: v_dual_add_f32 v107, v107, v111
	v_dual_add_f32 v116, v116, v120 :: v_dual_add_f32 v117, v117, v121
	v_dual_add_f32 v118, v118, v122 :: v_dual_add_f32 v119, v119, v123
	v_dual_add_f32 v128, v128, v132 :: v_dual_add_f32 v129, v129, v133
	v_dual_add_f32 v130, v130, v134 :: v_dual_add_f32 v131, v131, v135
	v_dual_add_f32 v140, v140, v144 :: v_dual_add_f32 v141, v141, v145
	v_dual_add_f32 v142, v142, v146 :: v_dual_add_f32 v143, v143, v147
	v_dual_add_f32 v152, v152, v156 :: v_dual_add_f32 v153, v153, v157
	v_dual_add_f32 v154, v154, v158 :: v_dual_add_f32 v155, v155, v159
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_dual_add_f32 v100, v100, v104 :: v_dual_add_f32 v101, v101, v105
	v_dual_add_f32 v102, v102, v106 :: v_dual_add_f32 v103, v103, v107
	v_dual_add_f32 v112, v112, v116 :: v_dual_add_f32 v113, v113, v117
	v_dual_add_f32 v114, v114, v118 :: v_dual_add_f32 v115, v115, v119
	v_dual_add_f32 v124, v124, v128 :: v_dual_add_f32 v125, v125, v129
	v_dual_add_f32 v126, v126, v130 :: v_dual_add_f32 v127, v127, v131
	v_dual_add_f32 v136, v136, v140 :: v_dual_add_f32 v137, v137, v141
	v_dual_add_f32 v138, v138, v142 :: v_dual_add_f32 v139, v139, v143
	v_dual_add_f32 v148, v148, v152 :: v_dual_add_f32 v149, v149, v153
	v_dual_add_f32 v150, v150, v154 :: v_dual_add_f32 v151, v151, v155
	ds_swizzle_b32 v160, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v161, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v162, v66 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v163, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v164, v76 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v165, v77 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v166, v78 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v167, v79 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v168, v88 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v169, v89 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v170, v90 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v171, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v172, v100 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v173, v101 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v174, v102 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v175, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v176, v112 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v177, v113 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v178, v114 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v179, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v180, v124 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v181, v125 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v182, v126 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v183, v127 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v184, v136 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v185, v137 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v186, v138 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v187, v139 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v188, v148 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v189, v149 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v190, v150 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v191, v151 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x1f
	v_add_f32_e32 v64, v64, v160
	s_wait_dscnt 0x1e
	v_add_f32_e32 v65, v65, v161
	s_wait_dscnt 0x1d
	v_add_f32_e32 v66, v66, v162
	s_wait_dscnt 0x1c
	v_add_f32_e32 v67, v67, v163
	s_wait_dscnt 0x1b
	v_add_f32_e32 v76, v76, v164
	s_wait_dscnt 0x1a
	v_add_f32_e32 v77, v77, v165
	s_wait_dscnt 0x19
	v_add_f32_e32 v78, v78, v166
	s_wait_dscnt 0x18
	v_add_f32_e32 v79, v79, v167
	s_wait_dscnt 0x17
	v_add_f32_e32 v88, v88, v168
	s_wait_dscnt 0x16
	v_add_f32_e32 v89, v89, v169
	s_wait_dscnt 0x15
	v_add_f32_e32 v90, v90, v170
	s_wait_dscnt 0x14
	v_add_f32_e32 v91, v91, v171
	s_wait_dscnt 0x13
	v_add_f32_e32 v100, v100, v172
	s_wait_dscnt 0x12
	v_add_f32_e32 v101, v101, v173
	s_wait_dscnt 0x11
	v_add_f32_e32 v102, v102, v174
	s_wait_dscnt 0x10
	v_add_f32_e32 v103, v103, v175
	s_wait_dscnt 0xf
	v_add_f32_e32 v112, v112, v176
	s_wait_dscnt 0xe
	v_add_f32_e32 v113, v113, v177
	s_wait_dscnt 0xd
	v_add_f32_e32 v114, v114, v178
	s_wait_dscnt 0xc
	v_add_f32_e32 v115, v115, v179
	s_wait_dscnt 0xb
	v_add_f32_e32 v124, v124, v180
	s_wait_dscnt 0xa
	v_add_f32_e32 v125, v125, v181
	s_wait_dscnt 0x9
	v_add_f32_e32 v126, v126, v182
	s_wait_dscnt 0x8
	v_add_f32_e32 v127, v127, v183
	s_wait_dscnt 0x7
	v_add_f32_e32 v136, v136, v184
	s_wait_dscnt 0x6
	v_add_f32_e32 v137, v137, v185
	s_wait_dscnt 0x5
	v_add_f32_e32 v138, v138, v186
	s_wait_dscnt 0x4
	v_add_f32_e32 v139, v139, v187
	s_wait_dscnt 0x3
	v_add_f32_e32 v148, v148, v188
	s_wait_dscnt 0x2
	v_add_f32_e32 v149, v149, v189
	s_wait_dscnt 0x1
	v_add_f32_e32 v150, v150, v190
	s_wait_dscnt 0x0
	v_add_f32_e32 v151, v151, v191
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v160, v3, v64
	ds_bpermute_b32 v161, v3, v65
	ds_bpermute_b32 v162, v3, v66
	ds_bpermute_b32 v163, v3, v67
	ds_bpermute_b32 v164, v3, v76
	ds_bpermute_b32 v165, v3, v77
	ds_bpermute_b32 v166, v3, v78
	ds_bpermute_b32 v167, v3, v79
	ds_bpermute_b32 v168, v3, v88
	ds_bpermute_b32 v169, v3, v89
	ds_bpermute_b32 v170, v3, v90
	ds_bpermute_b32 v171, v3, v91
	ds_bpermute_b32 v172, v3, v100
	ds_bpermute_b32 v173, v3, v101
	ds_bpermute_b32 v174, v3, v102
	ds_bpermute_b32 v175, v3, v103
	ds_bpermute_b32 v176, v3, v112
	ds_bpermute_b32 v177, v3, v113
	ds_bpermute_b32 v178, v3, v114
	ds_bpermute_b32 v179, v3, v115
	ds_bpermute_b32 v180, v3, v124
	ds_bpermute_b32 v181, v3, v125
	ds_bpermute_b32 v182, v3, v126
	ds_bpermute_b32 v183, v3, v127
	ds_bpermute_b32 v184, v3, v136
	ds_bpermute_b32 v185, v3, v137
	ds_bpermute_b32 v186, v3, v138
	ds_bpermute_b32 v187, v3, v139
	ds_bpermute_b32 v188, v3, v148
	ds_bpermute_b32 v189, v3, v149
	ds_bpermute_b32 v190, v3, v150
	ds_bpermute_b32 v191, v3, v151
	s_wait_dscnt 0x1f
	v_add_f32_e32 v64, v64, v160
	s_wait_dscnt 0x1e
	v_add_f32_e32 v65, v65, v161
	s_wait_dscnt 0x1d
	v_add_f32_e32 v66, v66, v162
	s_wait_dscnt 0x1c
	v_add_f32_e32 v67, v67, v163
	s_wait_dscnt 0x1b
	v_add_f32_e32 v76, v76, v164
	s_wait_dscnt 0x1a
	v_add_f32_e32 v77, v77, v165
	s_wait_dscnt 0x19
	v_add_f32_e32 v78, v78, v166
	s_wait_dscnt 0x18
	v_add_f32_e32 v79, v79, v167
	s_wait_dscnt 0x17
	v_add_f32_e32 v88, v88, v168
	s_wait_dscnt 0x16
	v_add_f32_e32 v89, v89, v169
	s_wait_dscnt 0x15
	v_add_f32_e32 v90, v90, v170
	s_wait_dscnt 0x14
	v_add_f32_e32 v91, v91, v171
	s_wait_dscnt 0x13
	v_add_f32_e32 v100, v100, v172
	s_wait_dscnt 0x12
	v_add_f32_e32 v101, v101, v173
	s_wait_dscnt 0x11
	v_add_f32_e32 v102, v102, v174
	s_wait_dscnt 0x10
	v_add_f32_e32 v103, v103, v175
	s_wait_dscnt 0xf
	v_add_f32_e32 v112, v112, v176
	s_wait_dscnt 0xe
	v_add_f32_e32 v113, v113, v177
	s_wait_dscnt 0xd
	v_add_f32_e32 v114, v114, v178
	s_wait_dscnt 0xc
	v_add_f32_e32 v115, v115, v179
	s_wait_dscnt 0xb
	v_add_f32_e32 v124, v124, v180
	s_wait_dscnt 0xa
	v_add_f32_e32 v125, v125, v181
	s_wait_dscnt 0x9
	v_add_f32_e32 v126, v126, v182
	s_wait_dscnt 0x8
	v_add_f32_e32 v127, v127, v183
	s_wait_dscnt 0x7
	v_add_f32_e32 v136, v136, v184
	s_wait_dscnt 0x6
	v_add_f32_e32 v137, v137, v185
	s_wait_dscnt 0x5
	v_add_f32_e32 v138, v138, v186
	s_wait_dscnt 0x4
	v_add_f32_e32 v139, v139, v187
	s_wait_dscnt 0x3
	v_add_f32_e32 v148, v148, v188
	s_wait_dscnt 0x2
	v_add_f32_e32 v149, v149, v189
	s_wait_dscnt 0x1
	v_add_f32_e32 v150, v150, v190
	s_wait_dscnt 0x0
	v_add_f32_e32 v151, v151, v191
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v160, v3, v64
	ds_bpermute_b32 v161, v3, v65
	ds_bpermute_b32 v162, v3, v66
	ds_bpermute_b32 v163, v3, v67
	ds_bpermute_b32 v164, v3, v76
	ds_bpermute_b32 v165, v3, v77
	ds_bpermute_b32 v166, v3, v78
	ds_bpermute_b32 v167, v3, v79
	ds_bpermute_b32 v168, v3, v88
	ds_bpermute_b32 v169, v3, v89
	ds_bpermute_b32 v170, v3, v90
	ds_bpermute_b32 v171, v3, v91
	ds_bpermute_b32 v172, v3, v100
	ds_bpermute_b32 v173, v3, v101
	ds_bpermute_b32 v174, v3, v102
	ds_bpermute_b32 v175, v3, v103
	ds_bpermute_b32 v176, v3, v112
	ds_bpermute_b32 v177, v3, v113
	ds_bpermute_b32 v178, v3, v114
	ds_bpermute_b32 v179, v3, v115
	ds_bpermute_b32 v180, v3, v124
	ds_bpermute_b32 v181, v3, v125
	ds_bpermute_b32 v182, v3, v126
	ds_bpermute_b32 v183, v3, v127
	ds_bpermute_b32 v184, v3, v136
	ds_bpermute_b32 v185, v3, v137
	ds_bpermute_b32 v186, v3, v138
	ds_bpermute_b32 v187, v3, v139
	ds_bpermute_b32 v188, v3, v148
	ds_bpermute_b32 v189, v3, v149
	ds_bpermute_b32 v190, v3, v150
	ds_bpermute_b32 v191, v3, v151
	s_wait_dscnt 0x1f
	v_add_f32_e32 v64, v64, v160
	s_wait_dscnt 0x1e
	v_add_f32_e32 v65, v65, v161
	s_wait_dscnt 0x1d
	v_add_f32_e32 v66, v66, v162
	s_wait_dscnt 0x1c
	v_add_f32_e32 v67, v67, v163
	s_wait_dscnt 0x1b
	v_add_f32_e32 v76, v76, v164
	s_wait_dscnt 0x1a
	v_add_f32_e32 v77, v77, v165
	s_wait_dscnt 0x19
	v_add_f32_e32 v78, v78, v166
	s_wait_dscnt 0x18
	v_add_f32_e32 v79, v79, v167
	s_wait_dscnt 0x17
	v_add_f32_e32 v88, v88, v168
	s_wait_dscnt 0x16
	v_add_f32_e32 v89, v89, v169
	s_wait_dscnt 0x15
	v_add_f32_e32 v90, v90, v170
	s_wait_dscnt 0x14
	v_add_f32_e32 v91, v91, v171
	s_wait_dscnt 0x13
	v_add_f32_e32 v100, v100, v172
	s_wait_dscnt 0x12
	v_add_f32_e32 v101, v101, v173
	s_wait_dscnt 0x11
	v_add_f32_e32 v102, v102, v174
	s_wait_dscnt 0x10
	v_add_f32_e32 v103, v103, v175
	s_wait_dscnt 0xf
	v_add_f32_e32 v112, v112, v176
	s_wait_dscnt 0xe
	v_add_f32_e32 v113, v113, v177
	s_wait_dscnt 0xd
	v_add_f32_e32 v114, v114, v178
	s_wait_dscnt 0xc
	v_add_f32_e32 v115, v115, v179
	s_wait_dscnt 0xb
	v_add_f32_e32 v124, v124, v180
	s_wait_dscnt 0xa
	v_add_f32_e32 v125, v125, v181
	s_wait_dscnt 0x9
	v_add_f32_e32 v126, v126, v182
	s_wait_dscnt 0x8
	v_add_f32_e32 v127, v127, v183
	s_wait_dscnt 0x7
	v_add_f32_e32 v136, v136, v184
	s_wait_dscnt 0x6
	v_add_f32_e32 v137, v137, v185
	s_wait_dscnt 0x5
	v_add_f32_e32 v138, v138, v186
	s_wait_dscnt 0x4
	v_add_f32_e32 v139, v139, v187
	s_wait_dscnt 0x3
	v_add_f32_e32 v148, v148, v188
	s_wait_dscnt 0x2
	v_add_f32_e32 v149, v149, v189
	s_wait_dscnt 0x1
	v_add_f32_e32 v150, v150, v190
	s_wait_dscnt 0x0
	v_add_f32_e32 v151, v151, v191
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v160, v3, v64
	ds_bpermute_b32 v161, v3, v65
	ds_bpermute_b32 v162, v3, v66
	ds_bpermute_b32 v163, v3, v67
	ds_bpermute_b32 v164, v3, v76
	ds_bpermute_b32 v165, v3, v77
	ds_bpermute_b32 v166, v3, v78
	ds_bpermute_b32 v167, v3, v79
	ds_bpermute_b32 v168, v3, v88
	ds_bpermute_b32 v169, v3, v89
	ds_bpermute_b32 v170, v3, v90
	ds_bpermute_b32 v171, v3, v91
	ds_bpermute_b32 v172, v3, v100
	ds_bpermute_b32 v173, v3, v101
	ds_bpermute_b32 v174, v3, v102
	ds_bpermute_b32 v175, v3, v103
	ds_bpermute_b32 v176, v3, v112
	ds_bpermute_b32 v177, v3, v113
	ds_bpermute_b32 v178, v3, v114
	ds_bpermute_b32 v179, v3, v115
	ds_bpermute_b32 v180, v3, v124
	ds_bpermute_b32 v181, v3, v125
	ds_bpermute_b32 v182, v3, v126
	ds_bpermute_b32 v183, v3, v127
	ds_bpermute_b32 v184, v3, v136
	ds_bpermute_b32 v185, v3, v137
	ds_bpermute_b32 v186, v3, v138
	ds_bpermute_b32 v187, v3, v139
	ds_bpermute_b32 v188, v3, v148
	ds_bpermute_b32 v189, v3, v149
	ds_bpermute_b32 v190, v3, v150
	ds_bpermute_b32 v191, v3, v151
	s_wait_dscnt 0x1f
	v_add_f32_e32 v64, v64, v160
	s_wait_dscnt 0x1e
	v_add_f32_e32 v65, v65, v161
	s_wait_dscnt 0x1d
	v_add_f32_e32 v66, v66, v162
	s_wait_dscnt 0x1c
	v_add_f32_e32 v67, v67, v163
	s_wait_dscnt 0x1b
	v_add_f32_e32 v76, v76, v164
	s_wait_dscnt 0x1a
	v_add_f32_e32 v77, v77, v165
	s_wait_dscnt 0x19
	v_add_f32_e32 v78, v78, v166
	s_wait_dscnt 0x18
	v_add_f32_e32 v79, v79, v167
	s_wait_dscnt 0x17
	v_add_f32_e32 v88, v88, v168
	s_wait_dscnt 0x16
	v_add_f32_e32 v89, v89, v169
	s_wait_dscnt 0x15
	v_add_f32_e32 v90, v90, v170
	s_wait_dscnt 0x14
	v_add_f32_e32 v91, v91, v171
	s_wait_dscnt 0x13
	v_add_f32_e32 v100, v100, v172
	s_wait_dscnt 0x12
	v_add_f32_e32 v101, v101, v173
	s_wait_dscnt 0x11
	v_add_f32_e32 v102, v102, v174
	s_wait_dscnt 0x10
	v_add_f32_e32 v103, v103, v175
	s_wait_dscnt 0xf
	v_add_f32_e32 v112, v112, v176
	s_wait_dscnt 0xe
	v_add_f32_e32 v113, v113, v177
	s_wait_dscnt 0xd
	v_add_f32_e32 v114, v114, v178
	s_wait_dscnt 0xc
	v_add_f32_e32 v115, v115, v179
	s_wait_dscnt 0xb
	v_add_f32_e32 v124, v124, v180
	s_wait_dscnt 0xa
	v_add_f32_e32 v125, v125, v181
	s_wait_dscnt 0x9
	v_add_f32_e32 v126, v126, v182
	s_wait_dscnt 0x8
	v_add_f32_e32 v127, v127, v183
	s_wait_dscnt 0x7
	v_add_f32_e32 v136, v136, v184
	s_wait_dscnt 0x6
	v_add_f32_e32 v137, v137, v185
	s_wait_dscnt 0x5
	v_add_f32_e32 v138, v138, v186
	s_wait_dscnt 0x4
	v_add_f32_e32 v139, v139, v187
	s_wait_dscnt 0x3
	v_add_f32_e32 v148, v148, v188
	s_wait_dscnt 0x2
	v_add_f32_e32 v149, v149, v189
	s_wait_dscnt 0x1
	v_add_f32_e32 v150, v150, v190
	s_wait_dscnt 0x0
	v_add_f32_e32 v151, v151, v191
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v160, v3, v64
	ds_bpermute_b32 v161, v3, v65
	ds_bpermute_b32 v162, v3, v66
	ds_bpermute_b32 v163, v3, v67
	ds_bpermute_b32 v164, v3, v76
	ds_bpermute_b32 v165, v3, v77
	ds_bpermute_b32 v166, v3, v78
	ds_bpermute_b32 v167, v3, v79
	ds_bpermute_b32 v168, v3, v88
	ds_bpermute_b32 v169, v3, v89
	ds_bpermute_b32 v170, v3, v90
	ds_bpermute_b32 v171, v3, v91
	ds_bpermute_b32 v172, v3, v100
	ds_bpermute_b32 v173, v3, v101
	ds_bpermute_b32 v174, v3, v102
	ds_bpermute_b32 v175, v3, v103
	ds_bpermute_b32 v176, v3, v112
	ds_bpermute_b32 v177, v3, v113
	ds_bpermute_b32 v178, v3, v114
	ds_bpermute_b32 v179, v3, v115
	ds_bpermute_b32 v180, v3, v124
	ds_bpermute_b32 v181, v3, v125
	ds_bpermute_b32 v182, v3, v126
	ds_bpermute_b32 v183, v3, v127
	ds_bpermute_b32 v184, v3, v136
	ds_bpermute_b32 v185, v3, v137
	ds_bpermute_b32 v186, v3, v138
	ds_bpermute_b32 v187, v3, v139
	ds_bpermute_b32 v188, v3, v148
	ds_bpermute_b32 v189, v3, v149
	ds_bpermute_b32 v190, v3, v150
	ds_bpermute_b32 v191, v3, v151
	s_wait_dscnt 0x1f
	v_add_f32_e32 v64, v64, v160
	s_wait_dscnt 0x1e
	v_add_f32_e32 v65, v65, v161
	s_wait_dscnt 0x1d
	v_add_f32_e32 v66, v66, v162
	s_wait_dscnt 0x1c
	v_add_f32_e32 v67, v67, v163
	s_wait_dscnt 0x1b
	v_add_f32_e32 v76, v76, v164
	s_wait_dscnt 0x1a
	v_add_f32_e32 v77, v77, v165
	s_wait_dscnt 0x19
	v_add_f32_e32 v78, v78, v166
	s_wait_dscnt 0x18
	v_add_f32_e32 v79, v79, v167
	s_wait_dscnt 0x17
	v_add_f32_e32 v88, v88, v168
	s_wait_dscnt 0x16
	v_add_f32_e32 v89, v89, v169
	s_wait_dscnt 0x15
	v_add_f32_e32 v90, v90, v170
	s_wait_dscnt 0x14
	v_add_f32_e32 v91, v91, v171
	s_wait_dscnt 0x13
	v_add_f32_e32 v100, v100, v172
	s_wait_dscnt 0x12
	v_add_f32_e32 v101, v101, v173
	s_wait_dscnt 0x11
	v_add_f32_e32 v102, v102, v174
	s_wait_dscnt 0x10
	v_add_f32_e32 v103, v103, v175
	s_wait_dscnt 0xf
	v_add_f32_e32 v112, v112, v176
	s_wait_dscnt 0xe
	v_add_f32_e32 v113, v113, v177
	s_wait_dscnt 0xd
	v_add_f32_e32 v114, v114, v178
	s_wait_dscnt 0xc
	v_add_f32_e32 v115, v115, v179
	s_wait_dscnt 0xb
	v_add_f32_e32 v124, v124, v180
	s_wait_dscnt 0xa
	v_add_f32_e32 v125, v125, v181
	s_wait_dscnt 0x9
	v_add_f32_e32 v126, v126, v182
	s_wait_dscnt 0x8
	v_add_f32_e32 v127, v127, v183
	s_wait_dscnt 0x7
	v_add_f32_e32 v136, v136, v184
	s_wait_dscnt 0x6
	v_add_f32_e32 v137, v137, v185
	s_wait_dscnt 0x5
	v_add_f32_e32 v138, v138, v186
	s_wait_dscnt 0x4
	v_add_f32_e32 v139, v139, v187
	s_wait_dscnt 0x3
	v_add_f32_e32 v148, v148, v188
	s_wait_dscnt 0x2
	v_add_f32_e32 v149, v149, v189
	s_wait_dscnt 0x1
	v_add_f32_e32 v150, v150, v190
	s_wait_dscnt 0x0
	v_add_f32_e32 v151, v151, v191
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v56, s46
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v57, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v58, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v59, s47
	s_mul_i32 s47, s45, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v60, s47
	s_mul_i32 s47, s45, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v61, s47
	s_mul_i32 s47, s45, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v62, s47
	s_mul_i32 s47, s45, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v63, s47
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b8_p0_single
	global_load_b32 v24, v56, s[8:9] offset:0
	global_load_b32 v25, v56, s[8:9] offset:4
	global_load_b32 v28, v57, s[8:9] offset:0
	global_load_b32 v29, v57, s[8:9] offset:4
	global_load_b32 v32, v58, s[8:9] offset:0
	global_load_b32 v33, v58, s[8:9] offset:4
	global_load_b32 v36, v59, s[8:9] offset:0
	global_load_b32 v37, v59, s[8:9] offset:4
	global_load_b32 v40, v60, s[8:9] offset:0
	global_load_b32 v41, v60, s[8:9] offset:4
	global_load_b32 v44, v61, s[8:9] offset:0
	global_load_b32 v45, v61, s[8:9] offset:4
	global_load_b32 v48, v62, s[8:9] offset:0
	global_load_b32 v49, v62, s[8:9] offset:4
	global_load_b32 v52, v63, s[8:9] offset:0
	global_load_b32 v53, v63, s[8:9] offset:4
	s_wait_loadcnt 0xf
	v_add_f32_e32 v24, v64, v24
	global_store_b32 v56, v24, s[8:9] offset:0
	s_wait_loadcnt 0xe
	v_add_f32_e32 v25, v65, v25
	global_store_b32 v56, v25, s[8:9] offset:4
	s_wait_loadcnt 0xd
	v_add_f32_e32 v28, v76, v28
	global_store_b32 v57, v28, s[8:9] offset:0
	s_wait_loadcnt 0xc
	v_add_f32_e32 v29, v77, v29
	global_store_b32 v57, v29, s[8:9] offset:4
	s_wait_loadcnt 0xb
	v_add_f32_e32 v32, v88, v32
	global_store_b32 v58, v32, s[8:9] offset:0
	s_wait_loadcnt 0xa
	v_add_f32_e32 v33, v89, v33
	global_store_b32 v58, v33, s[8:9] offset:4
	s_wait_loadcnt 0x9
	v_add_f32_e32 v36, v100, v36
	global_store_b32 v59, v36, s[8:9] offset:0
	s_wait_loadcnt 0x8
	v_add_f32_e32 v37, v101, v37
	global_store_b32 v59, v37, s[8:9] offset:4
	s_wait_loadcnt 0x7
	v_add_f32_e32 v40, v112, v40
	global_store_b32 v60, v40, s[8:9] offset:0
	s_wait_loadcnt 0x6
	v_add_f32_e32 v41, v113, v41
	global_store_b32 v60, v41, s[8:9] offset:4
	s_wait_loadcnt 0x5
	v_add_f32_e32 v44, v124, v44
	global_store_b32 v61, v44, s[8:9] offset:0
	s_wait_loadcnt 0x4
	v_add_f32_e32 v45, v125, v45
	global_store_b32 v61, v45, s[8:9] offset:4
	s_wait_loadcnt 0x3
	v_add_f32_e32 v48, v136, v48
	global_store_b32 v62, v48, s[8:9] offset:0
	s_wait_loadcnt 0x2
	v_add_f32_e32 v49, v137, v49
	global_store_b32 v62, v49, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v52, v148, v52
	global_store_b32 v63, v52, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v53, v149, v53
	global_store_b32 v63, v53, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b8_p0_next
	.Lrx_b8_p0_single:
	global_load_b32 v24, v56, s[8:9] offset:0
	global_load_b32 v28, v57, s[8:9] offset:0
	global_load_b32 v32, v58, s[8:9] offset:0
	global_load_b32 v36, v59, s[8:9] offset:0
	global_load_b32 v40, v60, s[8:9] offset:0
	global_load_b32 v44, v61, s[8:9] offset:0
	global_load_b32 v48, v62, s[8:9] offset:0
	global_load_b32 v52, v63, s[8:9] offset:0
	s_wait_loadcnt 0x7
	v_add_f32_e32 v24, v24, v64
	global_store_b32 v56, v24, s[8:9] offset:0
	s_wait_loadcnt 0x6
	v_add_f32_e32 v28, v28, v76
	global_store_b32 v57, v28, s[8:9] offset:0
	s_wait_loadcnt 0x5
	v_add_f32_e32 v32, v32, v88
	global_store_b32 v58, v32, s[8:9] offset:0
	s_wait_loadcnt 0x4
	v_add_f32_e32 v36, v36, v100
	global_store_b32 v59, v36, s[8:9] offset:0
	s_wait_loadcnt 0x3
	v_add_f32_e32 v40, v40, v112
	global_store_b32 v60, v40, s[8:9] offset:0
	s_wait_loadcnt 0x2
	v_add_f32_e32 v44, v44, v124
	global_store_b32 v61, v44, s[8:9] offset:0
	s_wait_loadcnt 0x1
	v_add_f32_e32 v48, v48, v136
	global_store_b32 v62, v48, s[8:9] offset:0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v52, v52, v148
	global_store_b32 v63, v52, s[8:9] offset:0
	s_wait_storecnt 0x0
	s_branch .Lrx_b8_stored
	.Lrx_b8_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b8_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b8_p1_single
	global_load_b32 v26, v56, s[8:9] offset:8
	global_load_b32 v27, v56, s[8:9] offset:12
	global_load_b32 v30, v57, s[8:9] offset:8
	global_load_b32 v31, v57, s[8:9] offset:12
	global_load_b32 v34, v58, s[8:9] offset:8
	global_load_b32 v35, v58, s[8:9] offset:12
	global_load_b32 v38, v59, s[8:9] offset:8
	global_load_b32 v39, v59, s[8:9] offset:12
	global_load_b32 v42, v60, s[8:9] offset:8
	global_load_b32 v43, v60, s[8:9] offset:12
	global_load_b32 v46, v61, s[8:9] offset:8
	global_load_b32 v47, v61, s[8:9] offset:12
	global_load_b32 v50, v62, s[8:9] offset:8
	global_load_b32 v51, v62, s[8:9] offset:12
	global_load_b32 v54, v63, s[8:9] offset:8
	global_load_b32 v55, v63, s[8:9] offset:12
	s_wait_loadcnt 0xf
	v_add_f32_e32 v26, v66, v26
	global_store_b32 v56, v26, s[8:9] offset:8
	s_wait_loadcnt 0xe
	v_add_f32_e32 v27, v67, v27
	global_store_b32 v56, v27, s[8:9] offset:12
	s_wait_loadcnt 0xd
	v_add_f32_e32 v30, v78, v30
	global_store_b32 v57, v30, s[8:9] offset:8
	s_wait_loadcnt 0xc
	v_add_f32_e32 v31, v79, v31
	global_store_b32 v57, v31, s[8:9] offset:12
	s_wait_loadcnt 0xb
	v_add_f32_e32 v34, v90, v34
	global_store_b32 v58, v34, s[8:9] offset:8
	s_wait_loadcnt 0xa
	v_add_f32_e32 v35, v91, v35
	global_store_b32 v58, v35, s[8:9] offset:12
	s_wait_loadcnt 0x9
	v_add_f32_e32 v38, v102, v38
	global_store_b32 v59, v38, s[8:9] offset:8
	s_wait_loadcnt 0x8
	v_add_f32_e32 v39, v103, v39
	global_store_b32 v59, v39, s[8:9] offset:12
	s_wait_loadcnt 0x7
	v_add_f32_e32 v42, v114, v42
	global_store_b32 v60, v42, s[8:9] offset:8
	s_wait_loadcnt 0x6
	v_add_f32_e32 v43, v115, v43
	global_store_b32 v60, v43, s[8:9] offset:12
	s_wait_loadcnt 0x5
	v_add_f32_e32 v46, v126, v46
	global_store_b32 v61, v46, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v47, v127, v47
	global_store_b32 v61, v47, s[8:9] offset:12
	s_wait_loadcnt 0x3
	v_add_f32_e32 v50, v138, v50
	global_store_b32 v62, v50, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v51, v139, v51
	global_store_b32 v62, v51, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v54, v150, v54
	global_store_b32 v63, v54, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v55, v151, v55
	global_store_b32 v63, v55, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b8_stored
	.Lrx_b8_p1_single:
	global_load_b32 v26, v56, s[8:9] offset:8
	global_load_b32 v30, v57, s[8:9] offset:8
	global_load_b32 v34, v58, s[8:9] offset:8
	global_load_b32 v38, v59, s[8:9] offset:8
	global_load_b32 v42, v60, s[8:9] offset:8
	global_load_b32 v46, v61, s[8:9] offset:8
	global_load_b32 v50, v62, s[8:9] offset:8
	global_load_b32 v54, v63, s[8:9] offset:8
	s_wait_loadcnt 0x7
	v_add_f32_e32 v26, v26, v66
	global_store_b32 v56, v26, s[8:9] offset:8
	s_wait_loadcnt 0x6
	v_add_f32_e32 v30, v30, v78
	global_store_b32 v57, v30, s[8:9] offset:8
	s_wait_loadcnt 0x5
	v_add_f32_e32 v34, v34, v90
	global_store_b32 v58, v34, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v38, v38, v102
	global_store_b32 v59, v38, s[8:9] offset:8
	s_wait_loadcnt 0x3
	v_add_f32_e32 v42, v42, v114
	global_store_b32 v60, v42, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v46, v46, v126
	global_store_b32 v61, v46, s[8:9] offset:8
	s_wait_loadcnt 0x1
	v_add_f32_e32 v50, v50, v138
	global_store_b32 v62, v50, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v54, v54, v150
	global_store_b32 v63, v54, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b8_stored
	.Lrx_b8_stored:
	s_branch .Lrx_end
	.Lrx_end:
	s_endpgm
.Lgemv_mq4g256v2_residual_xbatch_end:
.size gemv_mq4g256v2_residual_xbatch, .Lgemv_mq4g256v2_residual_xbatch_end-gemv_mq4g256v2_residual_xbatch
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_mq4g256v2_residual_xbatch
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 36
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
	.amdhsa_next_free_vgpr 252
	.amdhsa_next_free_sgpr 48
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_mq4g256v2_residual_xbatch_end-gemv_mq4g256v2_residual_xbatch)<<4)&4080)>>4
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
        .name: A
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: y
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 28
        .size: 4
        .value_kind: by_value
      - 
        .name: B
        .offset: 32
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .max_flat_workgroup_size: 32
    .name: gemv_mq4g256v2_residual_xbatch
    .private_segment_fixed_size: 0
    .sgpr_count: 50
    .sgpr_spill_count: 0
    .symbol: gemv_mq4g256v2_residual_xbatch.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 252
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
