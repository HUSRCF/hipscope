	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1010"
	.protected	magnum_butterfly_rotate_f32 ; -- Begin function magnum_butterfly_rotate_f32
	.globl	magnum_butterfly_rotate_f32
	.p2align	8
	.type	magnum_butterfly_rotate_f32,@function
magnum_butterfly_rotate_f32:            ; @magnum_butterfly_rotate_f32
; %bb.0:
	s_clause 0x1
	s_load_dword s0, s[4:5], 0x2c
	s_load_dword s1, s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	v_mad_u64_u32 v[0:1], s0, s6, s0, v[0:1]
	v_lshrrev_b32_e32 v1, 5, v0
	v_cmp_gt_u32_e32 vcc_lo, s1, v1
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_clause 0x1
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	s_load_dwordx2 s[12:13], s[4:5], 0x10
	v_mov_b32_e32 v1, 0
	v_and_b32_e32 v5, 1, v0
	v_lshlrev_b64 v[1:2], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v3, vcc_lo, s0, v1
	v_add_co_ci_u32_e32 v4, vcc_lo, s1, v2, vcc_lo
	s_load_dwordx8 s[4:11], s[12:13], 0x0
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	s_load_dwordx2 s[0:1], s[12:13], 0x20
	global_load_dword v3, v[3:4], off
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v5, s5, -s5, vcc_lo
	s_waitcnt vmcnt(0)
	ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,1)
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v4, v5, v4
	v_and_b32_e32 v5, 2, v0
	v_fmac_f32_e32 v4, s4, v3
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	ds_swizzle_b32 v3, v4 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v5, s7, -s7, vcc_lo
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v3, v5, v3
	v_and_b32_e32 v5, 4, v0
	v_fmac_f32_e32 v3, s6, v4
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v5, s9, -s9, vcc_lo
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v4, v5, v4
	v_and_b32_e32 v5, 8, v0
	v_and_b32_e32 v0, 16, v0
	v_fmac_f32_e32 v4, s8, v3
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	ds_swizzle_b32 v3, v4 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v5, s11, -s11, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	v_cndmask_b32_e64 v0, s1, -s1, vcc_lo
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v3, v5, v3
	v_fmac_f32_e32 v3, s10, v4
	ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,16)
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v4, v0, v4
	v_add_co_u32 v0, vcc_lo, s2, v1
	v_add_co_ci_u32_e32 v1, vcc_lo, s3, v2, vcc_lo
	v_fmac_f32_e32 v4, s0, v3
	global_store_dword v[0:1], v4, off
.LBB0_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel magnum_butterfly_rotate_f32
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 14
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	magnum_butterfly_rotate_f32, .Lfunc_end0-magnum_butterfly_rotate_f32
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 344
; NumSgprs: 16
; NumVgprs: 6
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 6
; Occupancy: 20
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	magnum_butterfly_rotate_inv_f32 ; -- Begin function magnum_butterfly_rotate_inv_f32
	.globl	magnum_butterfly_rotate_inv_f32
	.p2align	8
	.type	magnum_butterfly_rotate_inv_f32,@function
magnum_butterfly_rotate_inv_f32:        ; @magnum_butterfly_rotate_inv_f32
; %bb.0:
	s_clause 0x1
	s_load_dword s0, s[4:5], 0x2c
	s_load_dword s1, s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	v_mad_u64_u32 v[0:1], s0, s6, s0, v[0:1]
	v_lshrrev_b32_e32 v1, 5, v0
	v_cmp_gt_u32_e32 vcc_lo, s1, v1
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB1_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_mov_b32_e32 v1, 0
	s_load_dwordx2 s[4:5], s[4:5], 0x10
	v_and_b32_e32 v5, 16, v0
	v_lshlrev_b64 v[1:2], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v3, vcc_lo, s0, v1
	v_add_co_ci_u32_e32 v4, vcc_lo, s1, v2, vcc_lo
	s_load_dwordx2 s[0:1], s[4:5], 0x20
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	s_load_dwordx8 s[4:11], s[4:5], 0x0
	global_load_dword v3, v[3:4], off
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v5, -s1, s1, vcc_lo
	s_waitcnt vmcnt(0)
	ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,16)
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v4, v5, v4
	v_and_b32_e32 v5, 8, v0
	v_fmac_f32_e32 v4, s0, v3
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	ds_swizzle_b32 v3, v4 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v5, -s11, s11, vcc_lo
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v3, v5, v3
	v_and_b32_e32 v5, 4, v0
	v_fmac_f32_e32 v3, s10, v4
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v5, -s9, s9, vcc_lo
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v4, v5, v4
	v_and_b32_e32 v5, 2, v0
	v_and_b32_e32 v0, 1, v0
	v_fmac_f32_e32 v4, s8, v3
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	ds_swizzle_b32 v3, v4 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v5, -s7, s7, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	v_cndmask_b32_e64 v0, -s5, s5, vcc_lo
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v3, v5, v3
	v_fmac_f32_e32 v3, s6, v4
	ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,1)
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v4, v0, v4
	v_add_co_u32 v0, vcc_lo, s2, v1
	v_add_co_ci_u32_e32 v1, vcc_lo, s3, v2, vcc_lo
	v_fmac_f32_e32 v4, s4, v3
	global_store_dword v[0:1], v4, off
.LBB1_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel magnum_butterfly_rotate_inv_f32
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	magnum_butterfly_rotate_inv_f32, .Lfunc_end1-magnum_butterfly_rotate_inv_f32
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 340
; NumSgprs: 14
; NumVgprs: 6
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 6
; Occupancy: 20
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	magnum_butterfly_adaptive ; -- Begin function magnum_butterfly_adaptive
	.globl	magnum_butterfly_adaptive
	.p2align	8
	.type	magnum_butterfly_adaptive,@function
magnum_butterfly_adaptive:              ; @magnum_butterfly_adaptive
; %bb.0:
	s_clause 0x1
	s_load_dword s0, s[4:5], 0x34
	s_load_dword s1, s[4:5], 0x20
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	v_mad_u64_u32 v[0:1], s0, s6, s0, v[0:1]
	v_lshrrev_b32_e32 v3, 5, v0
	v_cmp_gt_u32_e32 vcc_lo, s1, v3
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB2_6
; %bb.1:
	s_load_dwordx8 s[0:7], s[4:5], 0x0
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[1:2], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v5, vcc_lo, s0, v1
	v_add_co_ci_u32_e32 v6, vcc_lo, s1, v2, vcc_lo
	global_load_dword v7, v[5:6], off
	global_load_ubyte v4, v3, s[6:7]
	s_load_dwordx4 s[8:11], s[4:5], 0x0
	v_and_b32_e32 v5, 1, v0
	v_and_b32_e32 v6, 2, v0
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v5, s9, -s9, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v6
	v_cndmask_b32_e64 v6, s11, -s11, vcc_lo
	s_waitcnt vmcnt(1)
	ds_swizzle_b32 v3, v7 offset:swizzle(SWAP,1)
	s_waitcnt vmcnt(0)
	v_cmp_ne_u16_e32 vcc_lo, 0, v4
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v5, v5, v3
	v_fmac_f32_e32 v5, s8, v7
	ds_swizzle_b32 v3, v5 offset:swizzle(SWAP,2)
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v3, v6, v3
	v_fmac_f32_e32 v3, s10, v5
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB2_3
; %bb.2:
	s_load_dwordx2 s[6:7], s[4:5], 0x10
	ds_swizzle_b32 v5, v3 offset:swizzle(SWAP,4)
	v_and_b32_e32 v6, 4, v0
	v_cmp_eq_u32_e32 vcc_lo, 0, v6
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v6, s7, -s7, vcc_lo
	v_mul_f32_e32 v5, v6, v5
	v_fmac_f32_e32 v5, s6, v3
	v_mov_b32_e32 v3, v5
.LBB2_3:
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_lt_u16_e32 vcc_lo, 1, v4
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB2_5
; %bb.4:
	s_load_dwordx4 s[4:7], s[4:5], 0x18
	ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,8)
	v_and_b32_e32 v5, 8, v0
	v_and_b32_e32 v0, 16, v0
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v5, s5, -s5, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	v_mul_f32_e32 v4, v5, v4
	v_cndmask_b32_e64 v0, s7, -s7, vcc_lo
	v_fmac_f32_e32 v4, s4, v3
	ds_swizzle_b32 v3, v4 offset:swizzle(SWAP,16)
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v3, v0, v3
	v_fmac_f32_e32 v3, s6, v4
.LBB2_5:
	s_or_b32 exec_lo, exec_lo, s0
	v_add_co_u32 v0, vcc_lo, s2, v1
	v_add_co_ci_u32_e32 v1, vcc_lo, s3, v2, vcc_lo
	s_waitcnt_vscnt null, 0x0
	global_store_dword v[0:1], v3, off
.LBB2_6:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel magnum_butterfly_adaptive
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 296
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end2:
	.size	magnum_butterfly_adaptive, .Lfunc_end2-magnum_butterfly_adaptive
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 400
; NumSgprs: 14
; NumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 8
; Occupancy: 20
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	magnum_butterfly_adaptive_inv ; -- Begin function magnum_butterfly_adaptive_inv
	.globl	magnum_butterfly_adaptive_inv
	.p2align	8
	.type	magnum_butterfly_adaptive_inv,@function
magnum_butterfly_adaptive_inv:          ; @magnum_butterfly_adaptive_inv
; %bb.0:
	s_clause 0x1
	s_load_dword s0, s[4:5], 0x34
	s_load_dword s1, s[4:5], 0x20
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	v_mad_u64_u32 v[0:1], s0, s6, s0, v[0:1]
	v_lshrrev_b32_e32 v5, 5, v0
	v_cmp_gt_u32_e32 vcc_lo, s1, v5
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB3_6
; %bb.1:
	s_load_dwordx8 s[0:7], s[4:5], 0x0
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[1:2], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v6, vcc_lo, s0, v1
	v_add_co_ci_u32_e32 v7, vcc_lo, s1, v2, vcc_lo
	global_load_ubyte v4, v5, s[6:7]
	global_load_dword v3, v[6:7], off
	s_clause 0x1
	s_load_dwordx4 s[8:11], s[4:5], 0x0
	s_load_dwordx2 s[0:1], s[4:5], 0x10
	s_waitcnt vmcnt(1)
	v_cmp_lt_u16_e32 vcc_lo, 1, v4
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB3_3
; %bb.2:
	s_load_dwordx4 s[12:15], s[4:5], 0x18
	s_waitcnt vmcnt(0)
	s_waitcnt_vscnt null, 0x0
	ds_swizzle_b32 v5, v3 offset:swizzle(SWAP,16)
	v_and_b32_e32 v6, 16, v0
	v_cmp_eq_u32_e32 vcc_lo, 0, v6
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v6, -s15, s15, vcc_lo
	v_mul_f32_e32 v5, v6, v5
	v_and_b32_e32 v6, 8, v0
	v_fmac_f32_e32 v5, s14, v3
	v_cmp_eq_u32_e32 vcc_lo, 0, v6
	ds_swizzle_b32 v3, v5 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v6, -s13, s13, vcc_lo
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v3, v6, v3
	v_fmac_f32_e32 v3, s12, v5
.LBB3_3:
	s_or_b32 exec_lo, exec_lo, s6
	v_cmp_ne_u16_e32 vcc_lo, 0, v4
	s_and_saveexec_b32 s4, vcc_lo
	s_cbranch_execz .LBB3_5
; %bb.4:
	s_waitcnt vmcnt(0)
	s_waitcnt_vscnt null, 0x0
	ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,4)
	v_and_b32_e32 v5, 4, v0
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v5, -s1, s1, vcc_lo
	v_mul_f32_e32 v4, v5, v4
	v_fmac_f32_e32 v4, s0, v3
	v_mov_b32_e32 v3, v4
.LBB3_5:
	s_or_b32 exec_lo, exec_lo, s4
	s_waitcnt vmcnt(0)
	s_waitcnt_vscnt null, 0x0
	ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,2)
	v_and_b32_e32 v5, 2, v0
	v_and_b32_e32 v0, 1, v0
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v5, -s11, s11, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	v_cndmask_b32_e64 v0, -s9, s9, vcc_lo
	v_mul_f32_e32 v4, v5, v4
	v_fmac_f32_e32 v4, s10, v3
	ds_swizzle_b32 v3, v4 offset:swizzle(SWAP,1)
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v3, v0, v3
	v_add_co_u32 v0, vcc_lo, s2, v1
	v_add_co_ci_u32_e32 v1, vcc_lo, s3, v2, vcc_lo
	v_fmac_f32_e32 v3, s8, v4
	global_store_dword v[0:1], v3, off
.LBB3_6:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel magnum_butterfly_adaptive_inv
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 296
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 16
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end3:
	.size	magnum_butterfly_adaptive_inv, .Lfunc_end3-magnum_butterfly_adaptive_inv
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 416
; NumSgprs: 18
; NumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 8
; Occupancy: 20
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 6, 3214868480
	.fill 48, 4, 3214868480
	.type	__hip_cuid_d5f203012c82e473,@object ; @__hip_cuid_d5f203012c82e473
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_d5f203012c82e473
__hip_cuid_d5f203012c82e473:
	.byte	0                               ; 0x0
	.size	__hip_cuid_d5f203012c82e473, 1

	.ident	"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.4 25012 e5bf7e55c91490b07c49d8960fa7983d864936c4)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_d5f203012c82e473
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         36
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         44
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         46
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         48
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         50
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         52
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         54
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         96
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           magnum_butterfly_rotate_f32
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         magnum_butterfly_rotate_f32.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         36
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         44
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         46
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         48
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         50
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         52
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         54
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         96
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           magnum_butterfly_rotate_inv_f32
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         magnum_butterfly_rotate_inv_f32.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         44
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         52
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         54
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         56
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         58
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         60
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         62
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         104
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           magnum_butterfly_adaptive
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         magnum_butterfly_adaptive.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         44
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         52
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         54
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         56
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         58
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         60
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         62
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         104
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           magnum_butterfly_adaptive_inv
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         magnum_butterfly_adaptive_inv.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1010
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
