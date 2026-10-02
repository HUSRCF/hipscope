use super::{Memory, Result, State};
use crate::{Inst, operand::{Operand, ImmField, VmemToken}};
fn width(name:&str)->Result<usize> {
    if name.contains("b256"){Ok(32)}else if name.contains("b128"){Ok(16)}else if name.contains("b96"){Ok(12)}else if name.contains("b64"){Ok(8)}else if name.contains("b32"){Ok(4)}else if name.contains("b16")||name.contains("u16"){Ok(2)}else if name.contains("u8"){Ok(1)}else{Err(format!("unknown memory width {name}"))}
}
pub(super) fn execute(name:&str,i:&Inst,s:&mut State,mem:&mut Memory,lds:&mut [u8])->Result<()> {
    if i.mods.ds.gds {return Err("GDS not modeled".into());}
    let ops=&i.operands;
    let n=width(name)?;
    if name.starts_with("s_load_") {
        let mut off=0u64;
        for op in &ops[2..] {off=off.wrapping_add(u64::from(s.read(op,0,0)?));}
        let addr=s.read64(&ops[1],0)?.wrapping_add(off);
        for w in 0..n/4 {let bytes=mem.read(addr+4*w as u64,4)?;s.put(&ops[0],0,w,u32::from_le_bytes(bytes.try_into().unwrap()))?;} return Ok(());
    }
    if name=="ds_swizzle_b32" {
        let pat=ops.iter().find_map(|o|if let Operand::Imm(ImmField::DsOffset(n))=o{Some(*n)}else{None}).unwrap_or(0);
        let mut values=[0;32];
        for (lane,val) in values.iter_mut().enumerate() {
            let src=if pat&0x8000!=0 {(lane&!3)|usize::from((pat>>(2*(lane&3)))&3)} else {((lane & usize::from(pat&31))|usize::from((pat>>5)&31)) ^ usize::from((pat>>10)&31)};
            *val=if src<32 && s.exec&(1<<src)!=0 {s.read(&ops[1],src,0)?}else{0};
        }
        for (lane,val) in values.into_iter().enumerate(){if s.exec&(1<<lane)!=0{s.put(&ops[0],lane,0,val)?;}}return Ok(());
    }
    let load=name.contains("load");
    let ds=name.starts_with("ds_");
    let buffer=name.starts_with("buffer_");
    let two=name.contains("2addr");
    for lane in 0..32 {
        if s.exec&(1<<lane)==0 {continue;}
        let mut offset=0i64;let mut off0=0usize;let mut off1=0usize;
        for op in ops {match op {Operand::Imm(ImmField::VmemOffset(n))=>offset+=i64::from(*n),Operand::Imm(ImmField::DsOffset(n))=>offset+=i64::from(*n),Operand::Imm(ImmField::DsOffset0(n))=>off0=usize::from(*n),Operand::Imm(ImmField::DsOffset1(n))=>off1=usize::from(*n),_=>()}}
        let mut addr;let mut raw_bounds=None;
        if ds {addr=u64::from(s.read(&ops[usize::from(load)],lane,0)?).wrapping_add_signed(offset);}
        else if buffer {
            let Operand::Reg(srd)=ops[2] else{return Err("buffer SRD missing".into());};
            if srd.len!=4{return Err("buffer SRD requires four registers".into());}
            let r=usize::from(srd.base);let word1=s.s[r+1];let word3=s.s[r+3];
            if word1>>16 != 0 {return Err("strided buffer descriptor unsupported".into());}
            // Raw byte-addressed SRDs: base[47:0], NUM_RECORDS in bytes.
            let base=u64::from(s.s[r])|(u64::from(word1&0xffff)<<32);
            let vaddr=if ops.iter().any(|o|matches!(o,Operand::Vmem(VmemToken::Offen))) {u64::from(s.read(&ops[1],lane,0)?)}else{0};
            let soff=u64::from(s.read(&ops[3],lane,0)?);
            let relative=vaddr.wrapping_add(soff).wrapping_add_signed(offset);
            if word3>>28&3!=3 {return Err("only raw buffer OOB mode 3 is supported".into());}
            raw_bounds=Some((relative,u64::from(s.s[r+2])));
            addr=base.wrapping_add(relative);
        } else if name.starts_with("global_") || name.starts_with("flat_") {
            let a=if load {1}else{0};addr=s.read64(&ops[a],lane)?.wrapping_add_signed(offset);
            if let Some(Operand::Reg(r))=ops.get(if load{2}else{2}) {if r.kind==crate::reg::Kind::S {addr=addr.wrapping_add(s.read64(&ops[2],lane)?);}}
        } else {return Err(format!("memory opcode {name} unsupported"));}
        let data=if load{&ops[0]}else if buffer{&ops[0]}else{&ops[1]};
        let count=if two{2}else{1};
        for part in 0..count {
            let cur=if two{addr+if part==0{off0}else{off1} as u64*n as u64}else{addr};
            let data_word=part*n.div_ceil(4);
            if load {
                let mut bytes=[0u8;16];
                for offset in (0..n).step_by(4) {
                    let size=(n-offset).min(4);
                    if raw_bounds.is_some_and(|(relative,limit)|relative.checked_add((offset+size)as u64).is_none_or(|end|end>limit)) {continue;}
                    let src=if ds {lds.get(cur as usize+offset..cur as usize+offset+size).ok_or_else(||format!("LDS read out of bounds {cur:#x}+{n}"))?}else{mem.read(cur+offset as u64,size)?};
                    bytes[offset..offset+size].copy_from_slice(src);
                }
                for w in 0..n.div_ceil(4) {let mut val=u32::from_le_bytes(bytes[w*4..w*4+4].try_into().unwrap());
                    if name.contains("d16_hi") {val=(s.read(data,lane,data_word+w)?&0xffff)|(val<<16);}
                    else if name.contains("d16") {val=(s.read(data,lane,data_word+w)?&0xffff0000)|(val&0xffff);}
                    s.put(data,lane,data_word+w,val)?;
                }
            } else {
                let mut bytes=[0u8;16];for w in 0..n.div_ceil(4) {let mut val=s.read(data,lane,data_word+w)?;if name.contains("d16_hi"){val>>=16;}bytes[w*4..w*4+4].copy_from_slice(&val.to_le_bytes());}
                for offset in (0..n).step_by(4) {
                    let size=(n-offset).min(4);
                    if raw_bounds.is_some_and(|(relative,limit)|relative.checked_add((offset+size)as u64).is_none_or(|end|end>limit)) {continue;}
                    if ds {lds.get_mut(cur as usize+offset..cur as usize+offset+size).ok_or_else(||format!("LDS write out of bounds {cur:#x}+{n}"))?.copy_from_slice(&bytes[offset..offset+size]);}else{mem.write(cur+offset as u64,&bytes[offset..offset+size])?;}
                }
            }
        }
    }Ok(())
}
