	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1010"
	.protected	magnum_dequant_hfq4     ; -- Begin function magnum_dequant_hfq4
	.globl	magnum_dequant_hfq4
	.p2align	8
	.type	magnum_dequant_hfq4,@function
magnum_dequant_hfq4:                    ; @magnum_dequant_hfq4
; %bb.0:
	s_clause 0x1
	s_load_dword s0, s[4:5], 0x34
	s_load_dword s1, s[4:5], 0x20
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	v_mad_u64_u32 v[0:1], s0, s6, s0, v[0:1]
	s_lshl_b32 s0, s1, 3
	v_lshrrev_b32_e32 v5, 5, v0
	v_cmp_gt_i32_e32 vcc_lo, s0, v5
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_6
; %bb.1:
	s_load_dwordx8 s[0:7], s[4:5], 0x0
	v_lshrrev_b32_e32 v1, 8, v0
	v_bfe_u32 v3, v0, 5, 3
	v_bfe_u32 v2, v0, 1, 4
	v_mul_u32_u24_e32 v8, 0x88, v1
	v_lshl_or_b32 v1, v3, 4, v2
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v2, s8, s0, v8
	v_add_co_ci_u32_e64 v4, s8, s1, 0, s8
	v_add_co_u32 v6, vcc_lo, v2, v1
	v_add_co_ci_u32_e32 v7, vcc_lo, 0, v4, vcc_lo
	s_clause 0x1
	global_load_ubyte v9, v[6:7], off offset:8
	global_load_dwordx2 v[1:2], v8, s[0:1]
	global_load_ubyte v4, v5, s[6:7]
	v_and_b32_e32 v5, 1, v0
	s_clause 0x1
	s_load_dwordx4 s[8:11], s[4:5], 0x0
	s_load_dwordx2 s[6:7], s[4:5], 0x10
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	s_waitcnt vmcnt(2)
	v_lshrrev_b32_e32 v6, 4, v9
	v_and_b32_e32 v7, 15, v9
	s_waitcnt vmcnt(0)
	v_cmp_lt_u16_e64 s0, 1, v4
	v_cndmask_b32_e32 v5, v6, v7, vcc_lo
	v_cvt_f32_ubyte0_e32 v5, v5
	v_fmac_f32_e32 v2, v1, v5
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_3
; %bb.2:
	s_load_dwordx4 s[12:15], s[4:5], 0x18
	s_waitcnt_vscnt null, 0x0
	ds_swizzle_b32 v1, v2 offset:swizzle(SWAP,16)
	v_and_b32_e32 v5, 16, v0
	v_cmp_eq_u32_e64 s0, 0, v5
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v5, -s15, s15, s0
	v_mul_f32_e32 v1, v5, v1
	v_and_b32_e32 v5, 8, v0
	v_fmac_f32_e32 v1, s14, v2
	v_cmp_eq_u32_e64 s0, 0, v5
	ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v5, -s13, s13, s0
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v2, v5, v2
	v_fmac_f32_e32 v2, s12, v1
.LBB0_3:
	s_or_b32 exec_lo, exec_lo, s1
	v_and_b32_e32 v1, 31, v0
	v_cmp_ne_u16_e64 s0, 0, v4
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_5
; %bb.4:
	s_waitcnt_vscnt null, 0x0
	ds_swizzle_b32 v4, v2 offset:swizzle(SWAP,4)
	v_and_b32_e32 v5, 4, v0
	v_cmp_eq_u32_e64 s0, 0, v5
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v5, -s7, s7, s0
	v_mul_f32_e32 v4, v5, v4
	v_fmac_f32_e32 v4, s6, v2
	v_mov_b32_e32 v2, v4
.LBB0_5:
	s_or_b32 exec_lo, exec_lo, s1
	s_waitcnt_vscnt null, 0x0
	ds_swizzle_b32 v4, v2 offset:swizzle(SWAP,2)
	v_and_b32_e32 v5, 2, v0
	v_and_b32_e32 v0, 0xffffff00, v0
	v_lshlrev_b32_e32 v3, 5, v3
	v_cmp_eq_u32_e64 s0, 0, v5
	v_or3_b32 v0, v3, v0, v1
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v3, -s9, s9, vcc_lo
	v_cndmask_b32_e64 v5, -s11, s11, s0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_mul_f32_e32 v4, v5, v4
	v_add_co_u32 v0, vcc_lo, s2, v0
	v_fmac_f32_e32 v4, s10, v2
	v_add_co_ci_u32_e32 v1, vcc_lo, s3, v1, vcc_lo
	ds_swizzle_b32 v2, v4 offset:swizzle(SWAP,1)
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v2, v3, v2
	v_fmac_f32_e32 v2, s8, v4
	global_store_dword v[0:1], v2, off
.LBB0_6:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel magnum_dequant_hfq4
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
		.amdhsa_next_free_vgpr 10
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
.Lfunc_end0:
	.size	magnum_dequant_hfq4, .Lfunc_end0-magnum_dequant_hfq4
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 544
; NumSgprs: 18
; NumVgprs: 10
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 10
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
	.protected	magnum_dequant_hfq4_norot ; -- Begin function magnum_dequant_hfq4_norot
	.globl	magnum_dequant_hfq4_norot
	.p2align	8
	.type	magnum_dequant_hfq4_norot,@function
magnum_dequant_hfq4_norot:              ; @magnum_dequant_hfq4_norot
; %bb.0:
	s_clause 0x1
	s_load_dword s0, s[4:5], 0x24
	s_load_dword s1, s[4:5], 0x10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	v_mad_u64_u32 v[0:1], s0, s6, s0, v[0:1]
	s_lshl_b32 s0, s1, 3
	v_lshrrev_b32_e32 v1, 5, v0
	v_cmp_gt_i32_e32 vcc_lo, s0, v1
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB1_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_lshrrev_b32_e32 v1, 8, v0
	v_bfe_u32 v5, v0, 5, 3
	v_bfe_u32 v2, v0, 1, 4
	v_mul_u32_u24_e32 v6, 0x88, v1
	v_lshl_or_b32 v1, v5, 4, v2
	v_lshlrev_b32_e32 v5, 5, v5
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v2, s4, s0, v6
	v_add_co_ci_u32_e64 v3, s4, s1, 0, s4
	v_add_co_u32 v1, vcc_lo, v2, v1
	v_add_co_ci_u32_e32 v2, vcc_lo, 0, v3, vcc_lo
	s_clause 0x1
	global_load_ubyte v7, v[1:2], off offset:8
	global_load_dwordx2 v[3:4], v6, s[0:1]
	v_and_b32_e32 v1, 31, v0
	v_and_b32_e32 v2, 0xffffff00, v0
	v_and_b32_e32 v6, 1, v0
	v_or3_b32 v0, v5, v2, v1
	v_cmp_eq_u32_e32 vcc_lo, 0, v6
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_waitcnt vmcnt(1)
	v_lshrrev_b32_e32 v8, 4, v7
	v_and_b32_e32 v7, 15, v7
	v_cndmask_b32_e32 v2, v8, v7, vcc_lo
	v_add_co_u32 v0, vcc_lo, s2, v0
	v_add_co_ci_u32_e32 v1, vcc_lo, s3, v1, vcc_lo
	v_cvt_f32_ubyte0_e32 v2, v2
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v4, v3, v2
	global_store_dword v[0:1], v4, off
.LBB1_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel magnum_dequant_hfq4_norot
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
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
		.amdhsa_next_free_vgpr 9
		.amdhsa_next_free_sgpr 7
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
	.size	magnum_dequant_hfq4_norot, .Lfunc_end1-magnum_dequant_hfq4_norot
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 256
; NumSgprs: 9
; NumVgprs: 9
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 9
; NumVGPRsForWavesPerEU: 9
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
	.type	__hip_cuid_34e3a69da3412ea0,@object ; @__hip_cuid_34e3a69da3412ea0
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_34e3a69da3412ea0
__hip_cuid_34e3a69da3412ea0:
	.byte	0                               ; 0x0
	.size	__hip_cuid_34e3a69da3412ea0, 1

	.ident	"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.4 25012 e5bf7e55c91490b07c49d8960fa7983d864936c4)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_34e3a69da3412ea0
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
    .name:           magnum_dequant_hfq4
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         magnum_dequant_hfq4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     10
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
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           magnum_dequant_hfq4_norot
    .private_segment_fixed_size: 0
    .sgpr_count:     9
    .sgpr_spill_count: 0
    .symbol:         magnum_dequant_hfq4_norot.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     9
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1010
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
