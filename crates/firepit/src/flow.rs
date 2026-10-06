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
    pub name: WeightName,
    pub members: Vec<Projection>,
    pub format: MatrixFormat,
    pub cols: usize,
    pub padding: usize,
    pub gemms: usize,
}

impl Gemv {
    fn new(model: &Model, op: usize, layer: u16, name: WeightName, out: Slot) -> Self {
        let matrix = weight(model, layer as usize, name);
        let (members, padding, gemms) = match out {
            Slot::Projection(layout) => {
                let layout = &model.projection_layouts[layout];
                (
                    layout.members().collect(),
                    layout.padding(),
                    layout.gemm_rows().len(),
                )
            }
            _ => (Vec::new(), 0, 1),
        };
        Self {
            op,
            name,
            members,
            format: matrix.format(),
            cols: matrix.cols(),
            padding,
            gemms,
        }
    }

    fn label(&self) -> String {
        if self.members.is_empty() {
            return format!("{:?}", self.name);
        }
        let names: Vec<_> = self.members.iter().map(|m| format!("{m:?}")).collect();
        format!("[{}]", names.join("|"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GemvInput {
    pub value: usize,
    pub gemvs: Vec<Gemv>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Analysis {
    pub values: Vec<Value>,
    pub unproduced: Vec<(usize, Buffer)>,
    pub gemv_inputs: Vec<GemvInput>,
}

fn group_by<K: PartialEq, T>(
    items: impl IntoIterator<Item = T>,
    key: impl Fn(&T) -> K,
) -> Vec<(K, Vec<T>)> {
    let mut groups: Vec<(K, Vec<T>)> = Vec::new();
    for item in items {
        let found = key(&item);
        match groups.iter_mut().find(|(seen, _)| *seen == found) {
            Some((_, group)) => group.push(item),
            None => groups.push((found, vec![item])),
        }
    }
    groups
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
    let mut gemvs = Vec::new();
    for (op, node) in nodes.iter().enumerate() {
        let (mut input, mut matrix, mut out) = (None, None, None);
        for usage in node.uses.iter().filter(|usage| !usage.write) {
            if let Buffer::Matrix { layer, name } = usage.buffer {
                matrix = Some((layer, name));
            }
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
            if let Buffer::Slot(slot) = usage.buffer {
                out = Some(slot);
            }
            let location = location(model, usage.buffer);
            values.push(Value {
                location,
                producer: op,
                consumers: Vec::new(),
            });
            latest.insert(location, values.len() - 1);
        }
        if let ("gemv", Some(value), Some((layer, name)), Some(out)) =
            (node.kind, input, matrix, out)
        {
            gemvs.push((value, Gemv::new(model, op, layer, name, out)));
        }
    }
    let gemv_inputs = group_by(gemvs, |(value, _)| *value)
        .into_iter()
        .map(|(value, gemvs)| GemvInput {
            value,
            gemvs: gemvs.into_iter().map(|(_, gemv)| gemv).collect(),
        })
        .collect();
    Analysis {
        values,
        unproduced,
        gemv_inputs,
    }
}

impl Analysis {
    pub fn fused(&self) -> impl Iterator<Item = &Gemv> {
        self.gemv_inputs
            .iter()
            .flat_map(|group| &group.gemvs)
            .filter(|gemv| gemv.members.len() > 1)
    }

    pub fn candidates(&self) -> Vec<Vec<&Gemv>> {
        self.gemv_inputs
            .iter()
            .flat_map(|group| group_by(&group.gemvs, |gemv| (gemv.format, gemv.cols)))
            .map(|(_, same)| same)
            .filter(|same| same.len() > 1)
            .collect()
    }
}

fn describe_location(model: &Model, location: Location) -> String {
    match location {
        Location::Slot(slot) => format!("{slot:?}"),
        Location::Other(buffer) => describe(model, buffer),
    }
}

fn describe(model: &Model, buffer: Buffer) -> String {
    match buffer {
        Buffer::Slot(slot) => format!("{slot:?}"),
        Buffer::Member { member, .. } => format!(
            "{}.{member:?}",
            describe_location(model, location(model, buffer))
        ),
        Buffer::Token => "Token".into(),
        Buffer::Position(_) => "Position".into(),
        Buffer::Cos => "Cos".into(),
        Buffer::Sin => "Sin".into(),
        Buffer::Selection { .. } => "Selection".into(),
        Buffer::State { .. } => "State".into(),
        Buffer::Matrix { name, .. } => format!("Matrix {name:?}"),
    }
}

fn section(out: &mut String, title: &str, lines: impl IntoIterator<Item = String>) {
    let _ = writeln!(out, "== {title}");
    let folded = group_by(lines, String::clone);
    if folded.is_empty() {
        out.push_str("none\n");
    }
    for (line, same) in folded {
        let _ = writeln!(out, "{line} x{}", same.len());
    }
}

pub fn report(model: &Model, nodes: &[OpNode]) -> String {
    let analysis = analyze(model, nodes);
    let kind = |op: usize| nodes[op].kind;
    let mut out = String::new();
    out.push_str(
        "# Norm Q and OutputGate both read AttnQGate whole: each uses half, so it counts twice.\n",
    );
    out.push_str("# Slots reused across layers keep the previous layer's value: a misordered layer k >= 1 shows as an edge to the old value, not as a read before any write.\n");
    section(
        &mut out,
        "same-input GEMVs",
        analysis.gemv_inputs.iter().map(|group| {
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
                    gemv.label(),
                    gemv.format,
                    gemv.cols
                );
                if gemv.padding > 0 {
                    let _ = write!(
                        line,
                        " (padding {} rows, {} prefill GEMMs)",
                        gemv.padding, gemv.gemms
                    );
                }
                line.push(';');
            }
            let formats = group_by(&group.gemvs, |gemv| gemv.format);
            if formats.len() > 1 {
                let formats: Vec<_> = formats.iter().map(|(format, _)| format).collect();
                let _ = write!(line, " formats differ: {formats:?};");
            }
            line
        }),
    );
    section(
        &mut out,
        "same-input, same-format GEMVs left apart",
        analysis.candidates().into_iter().map(|same| {
            let labels: Vec<_> = same.iter().map(|gemv| gemv.label()).collect();
            format!(
                "{} {:?} cols {}",
                labels.join(" + "),
                same[0].format,
                same[0].cols
            )
        }),
    );
    section(
        &mut out,
        "producer -> consumers",
        analysis.values.iter().map(|value| {
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
        }),
    );
    section(
        &mut out,
        "reads before any write",
        analysis
            .unproduced
            .iter()
            .map(|&(op, buffer)| format!("{} reads {}", kind(op), describe(model, buffer))),
    );
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
