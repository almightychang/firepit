use crate::model::{Model, Projection};
use crate::plan::{Slot, WeightName};
use crate::store::{Buffer, OpNode, weight};
use firepit_kernels::weights::MatrixFormat;
use std::collections::HashMap;
use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Location {
    Slot(Slot),
    Other(Buffer),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value {
    pub location: Location,
    pub producer: usize,
    pub consumers: Vec<(usize, Buffer)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemv {
    pub op: usize,
    pub layer: u16,
    pub name: WeightName,
    pub out: Slot,
    pub format: MatrixFormat,
    pub cols: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GemvInput {
    pub value: usize,
    pub gemvs: Vec<Gemv>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fused {
    pub members: Vec<Projection>,
    pub format: MatrixFormat,
    pub padding: usize,
    pub gemms: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Analysis {
    pub values: Vec<Value>,
    pub unproduced: Vec<(usize, Buffer)>,
    pub gemv_inputs: Vec<GemvInput>,
}

fn location(model: &Model, buffer: Buffer) -> Location {
    match buffer {
        Buffer::Slot(slot) => Location::Slot(slot),
        Buffer::Member { layer, member } => Location::Slot(
            Slot::projection(model, layer as usize, member)
                .expect("members in the flow belong to the model")
                .0,
        ),
        other => Location::Other(other),
    }
}

fn produced_elsewhere(buffer: Buffer) -> bool {
    matches!(
        buffer,
        Buffer::Token
            | Buffer::Position(_)
            | Buffer::Cos
            | Buffer::Sin
            | Buffer::State { .. }
            | Buffer::Matrix { .. }
    )
}

pub fn analyze(model: &Model, nodes: &[OpNode]) -> Analysis {
    let mut values: Vec<Value> = Vec::new();
    let mut latest = HashMap::<Location, usize>::new();
    let mut unproduced = Vec::new();
    let mut gemv_inputs: Vec<GemvInput> = Vec::new();
    for (op, node) in nodes.iter().enumerate() {
        let mut input = None;
        for usage in node.uses.iter().filter(|usage| !usage.write) {
            match latest.get(&location(model, usage.buffer)) {
                Some(&value) => {
                    values[value].consumers.push((op, usage.buffer));
                    if matches!(usage.buffer, Buffer::Slot(_) | Buffer::Member { .. }) {
                        input = Some(value);
                    }
                }
                None if !produced_elsewhere(usage.buffer) => unproduced.push((op, usage.buffer)),
                None => {}
            }
        }
        for usage in node.uses.iter().filter(|usage| usage.write) {
            let location = location(model, usage.buffer);
            values.push(Value {
                location,
                producer: op,
                consumers: Vec::new(),
            });
            latest.insert(location, values.len() - 1);
        }
        if node.kind != "gemv" {
            continue;
        }
        let matrix = node.uses.iter().find_map(|usage| match usage.buffer {
            Buffer::Matrix { layer, name } => Some((layer, name)),
            _ => None,
        });
        let out = node.uses.iter().find_map(|usage| match usage.buffer {
            Buffer::Slot(slot) if usage.write => Some(slot),
            _ => None,
        });
        let (Some(value), Some((layer, name)), Some(out)) = (input, matrix, out) else {
            continue;
        };
        let matrix = weight(model, layer as usize, name);
        let gemv = Gemv {
            op,
            layer,
            name,
            out,
            format: matrix.format(),
            cols: matrix.cols(),
        };
        match gemv_inputs.iter_mut().find(|group| group.value == value) {
            Some(group) => group.gemvs.push(gemv),
            None => gemv_inputs.push(GemvInput {
                value,
                gemvs: vec![gemv],
            }),
        }
    }
    Analysis {
        values,
        unproduced,
        gemv_inputs,
    }
}

impl Analysis {
    pub fn fused<'a>(&'a self, model: &'a Model) -> impl Iterator<Item = Fused> + 'a {
        self.gemv_inputs
            .iter()
            .flat_map(|group| &group.gemvs)
            .filter_map(move |gemv| {
                let Slot::Projection(layout) = gemv.out else {
                    return None;
                };
                let layout = &model.projection_layouts[layout];
                let members: Vec<_> = layout.members().collect();
                (members.len() > 1).then(|| Fused {
                    members,
                    format: gemv.format,
                    padding: layout.padding(),
                    gemms: layout.gemm_rows().len(),
                })
            })
    }

    pub fn candidates(&self) -> Vec<Vec<&Gemv>> {
        self.gemv_inputs
            .iter()
            .flat_map(|group| {
                let mut same: Vec<Vec<&Gemv>> = Vec::new();
                for gemv in &group.gemvs {
                    match same.iter_mut().find(|bucket| {
                        (bucket[0].format, bucket[0].cols) == (gemv.format, gemv.cols)
                    }) {
                        Some(bucket) => bucket.push(gemv),
                        None => same.push(vec![gemv]),
                    }
                }
                same.into_iter().filter(|bucket| bucket.len() > 1)
            })
            .collect()
    }
}

fn describe(model: &Model, buffer: Buffer) -> String {
    match buffer {
        Buffer::Slot(slot) => format!("{slot:?}"),
        Buffer::Member { layer, member } => {
            let (slot, _) = Slot::projection(model, layer as usize, member)
                .expect("members in the flow belong to the model");
            format!("{slot:?}.{member:?}")
        }
        Buffer::Token => "Token".into(),
        Buffer::Position(_) => "Position".into(),
        Buffer::Cos => "Cos".into(),
        Buffer::Sin => "Sin".into(),
        Buffer::Selection { .. } => "Selection".into(),
        Buffer::State { .. } => "State".into(),
        Buffer::Matrix { name, .. } => format!("Matrix {name:?}"),
    }
}

fn describe_location(model: &Model, location: Location) -> String {
    match location {
        Location::Slot(slot) => format!("{slot:?}"),
        Location::Other(buffer) => describe(model, buffer),
    }
}

fn fold(lines: impl IntoIterator<Item = String>) -> Vec<(String, usize)> {
    let mut folded: Vec<(String, usize)> = Vec::new();
    for line in lines {
        match folded.iter_mut().find(|(seen, _)| *seen == line) {
            Some((_, count)) => *count += 1,
            None => folded.push((line, 1)),
        }
    }
    folded
}

fn members(model: &Model, gemv: &Gemv) -> String {
    match gemv.out {
        Slot::Projection(layout) => {
            let names: Vec<_> = model.projection_layouts[layout]
                .members()
                .map(|member| format!("{member:?}"))
                .collect();
            format!("[{}]", names.join("|"))
        }
        _ => format!("{:?}", gemv.name),
    }
}

pub fn report(model: &Model, nodes: &[OpNode]) -> String {
    let analysis = analyze(model, nodes);
    let mut out = String::new();
    let kind = |op: usize| nodes[op].kind;
    out.push_str(
        "# Norm Q and OutputGate both read AttnQGate whole: each uses half, so it counts twice.\n",
    );
    out.push_str("# Slots reused across layers keep the previous layer's value: a misordered layer k >= 1 shows as an edge to the old value, not as a read before any write.\n");

    out.push_str("== horizontal concatenation\n");
    let lines = analysis.gemv_inputs.iter().map(|group| {
        let input = &analysis.values[group.value];
        let mut line = format!(
            "input {} <- {}:",
            describe_location(model, input.location),
            kind(input.producer)
        );
        for gemv in &group.gemvs {
            let _ = write!(
                line,
                " {} {:?} cols {}",
                members(model, gemv),
                gemv.format,
                gemv.cols
            );
            if let Slot::Projection(layout) = gemv.out {
                let layout = &model.projection_layouts[layout];
                if layout.padding() > 0 {
                    let _ = write!(
                        line,
                        " (padding {} rows, {} prefill GEMMs)",
                        layout.padding(),
                        layout.gemm_rows().len()
                    );
                }
            }
            line.push(';');
        }
        let mut formats: Vec<MatrixFormat> = Vec::new();
        for gemv in &group.gemvs {
            if !formats.contains(&gemv.format) {
                formats.push(gemv.format);
            }
        }
        if formats.len() > 1 {
            let _ = write!(line, " formats differ: {formats:?};");
        }
        line
    });
    for (line, count) in fold(lines) {
        let _ = writeln!(out, "{line} x{count}");
    }
    let candidates = analysis.candidates();
    if candidates.is_empty() {
        out.push_str("candidates: none\n");
    }
    for bucket in fold(candidates.iter().map(|bucket| {
        let names: Vec<_> = bucket.iter().map(|gemv| members(model, gemv)).collect();
        format!(
            "candidate {} {:?} cols {}",
            names.join(" + "),
            bucket[0].format,
            bucket[0].cols
        )
    })) {
        let _ = writeln!(out, "{} x{}", bucket.0, bucket.1);
    }

    out.push_str("== production -> consumption\n");
    let lines = analysis.values.iter().map(|value| {
        let consumers: Vec<_> = value
            .consumers
            .iter()
            .map(|&(op, buffer)| format!("{}({})", kind(op), describe(model, buffer)))
            .collect();
        format!(
            "{} -> {} -> [{}] ({})",
            kind(value.producer),
            describe_location(model, value.location),
            consumers.join(", "),
            consumers.len()
        )
    });
    for (line, count) in fold(lines) {
        let _ = writeln!(out, "{line} x{count}");
    }

    out.push_str("== reads before any write\n");
    if analysis.unproduced.is_empty() {
        out.push_str("none\n");
    }
    let lines = analysis
        .unproduced
        .iter()
        .map(|&(op, buffer)| format!("{} reads {}", kind(op), describe(model, buffer)));
    for (line, count) in fold(lines) {
        let _ = writeln!(out, "{line} x{count}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use crate::execute::run;
    use crate::plan::{Op, StepPlan};
    use crate::session::SessionId;
    use crate::store::OpNode;
    use cuda_core::Device;
    use firepit_kernels::sink::Dry;
    use std::sync::Arc;

    fn swapped_flow(layer: usize) -> crate::Result<(Arc<Model>, Vec<OpNode>)> {
        let device = Device::new(0)?;
        let stream = device.new_stream()?;
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/small-qwen.json"))?;
        let model = Arc::new(Model::random_from_config(
            &config["text_config"],
            8,
            7,
            &stream,
        )?);
        let engine = Engine::new(model.clone(), &device)?;
        let mut session = engine.new_session(SessionId(1))?;
        let mut plan = StepPlan::new(&model, 1, engine.tiles(), &stream)?;
        let position = |found: fn(&Op) -> bool| {
            plan.ops
                .iter()
                .position(|op| found(op) && op.layer() == Some(layer))
        };
        let projection = position(|op| {
            matches!(
                op,
                Op::Gemv {
                    name: WeightName::MoeProjection(_),
                    ..
                }
            )
        })
        .ok_or("no MoE projection")?;
        let top_k = position(|op| matches!(op, Op::TopK { .. })).ok_or("no top-k")?;
        plan.ops.swap(projection, top_k);
        let mut sessions = [&mut session];
        let (ops, tiles, mut ctx) = plan.context(&model, &mut sessions);
        ctx.record();
        run(ops, tiles, &mut ctx, &mut Dry)?;
        Ok((model.clone(), ctx.nodes()))
    }

    fn top_k(nodes: &[OpNode], layer: u16) -> usize {
        nodes
            .iter()
            .position(|node| node.kind == "moe_top_k" && node.layer == Some(layer))
            .expect("top-k node")
    }

    #[test]
    fn top_k_before_the_first_router_reads_unwritten_members() -> crate::Result<()> {
        let (model, nodes) = swapped_flow(0)?;
        let analysis = analyze(&model, &nodes);
        let op = top_k(&nodes, 0);
        let read: Vec<_> = analysis
            .unproduced
            .iter()
            .filter(|(at, _)| *at == op)
            .collect();
        assert_eq!(
            read.iter().map(|(_, buffer)| *buffer).collect::<Vec<_>>(),
            [
                Buffer::Member {
                    layer: 0,
                    member: Projection::Router
                },
                Buffer::Member {
                    layer: 0,
                    member: Projection::SharedGate
                },
            ]
        );
        assert_eq!(read.len(), analysis.unproduced.len());
        Ok(())
    }

    #[test]
    fn top_k_before_a_later_router_reads_the_previous_layer() -> crate::Result<()> {
        let (model, nodes) = swapped_flow(1)?;
        let analysis = analyze(&model, &nodes);
        assert!(analysis.unproduced.is_empty());
        let op = top_k(&nodes, 1);
        let producers: Vec<_> = analysis
            .values
            .iter()
            .filter(|value| value.consumers.iter().any(|(at, _)| *at == op))
            .filter(|value| matches!(value.location, Location::Slot(Slot::Projection(_))))
            .map(|value| nodes[value.producer].layer)
            .collect();
        assert_eq!(producers, [Some(0)]);
        Ok(())
    }
}
