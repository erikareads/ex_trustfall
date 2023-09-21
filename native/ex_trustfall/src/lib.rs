use rustler::*;
use rustler_elixir_fun;
use rustler_stored_term::StoredTerm;
use std::any::type_name;
use std::fmt::Debug;
use std::{collections::BTreeMap, sync::Arc};
use trustfall_core::{
    frontend::parse,
    interpreter::{
        execution::interpret_ir, Adapter, ContextIterator as BaseContextIterator,
        ContextOutcomeIterator, DataContext, ResolveEdgeInfo, ResolveInfo, VertexIterator,
    },
    ir::{EdgeParameters, FieldValue},
};

mod atoms {
    rustler::atoms! {
        ok,
        error,
    }
}

#[rustler::nif]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

fn load(_env: Env, _info: Term) -> bool {
    true
}

#[derive(NifStruct, Debug)]
#[module = "ExTrustfall.AdapterShim"]
struct AdapterShim<'a> {
    elixir_invoker: Term<'a>,
    resolve_starting_vertices: Term<'a>,
    resolve_property: Term<'a>,
    resolve_neighbors: Term<'a>,
    resolve_coercion: Term<'a>,
}

pub struct Context(DataContext<Arc<StoredTerm>>);

impl Context {
    fn active_vertex(&self) -> Result<Option<StoredTerm>, ()> {
        Ok(self.0.active_vertex().map(|arc| (**arc).clone()))
    }
}

pub struct ContextIterator(VertexIterator<'static, Context>);

impl ContextIterator {
    fn new(inner: VertexIterator<'static, DataContext<Arc<StoredTerm>>>) -> Self {
        Self(Box::new(inner.map(Context)))
    }
}

struct AdapterConfig<'a> {
    elixir_invoker: Term<'a>,
    resolve_starting_vertices: Term<'a>,
    resolve_property: Term<'a>,
    resolve_neighbors: Term<'a>,
    resolve_coercion: Term<'a>,
    env: Env<'a>,
}

impl<'a> AdapterConfig<'a> {
    pub fn new<'b>(adapter_shim: AdapterShim<'a>, env: Env<'a>) -> Result<Self, ()> {
        let elixir_invoker = adapter_shim.elixir_invoker;
        let resolve_starting_vertices = adapter_shim.resolve_starting_vertices;
        let resolve_property = adapter_shim.resolve_property;
        let resolve_neighbors = adapter_shim.resolve_neighbors;
        let resolve_coercion = adapter_shim.resolve_coercion;

        Ok(Self {
            elixir_invoker,
            resolve_starting_vertices,
            resolve_property,
            resolve_neighbors,
            resolve_coercion,
            env,
        })
    }
}

fn make_elixir_value<'a>(env: Env<'a>, value: FieldValue) -> Term<'a> {
    match value {
        FieldValue::Null => rustler::types::atom::nil().encode(env),
        FieldValue::Uint64(x) => x.encode(env),
        FieldValue::Int64(x) => x.encode(env),
        FieldValue::Float64(x) => x.encode(env),
        FieldValue::String(x) => x.encode(env),
        FieldValue::Boolean(x) => x.encode(env),
        FieldValue::DateTimeUtc(_) => todo!(),
        FieldValue::Enum(_) => todo!(),
        FieldValue::List(x) => x
            .into_iter()
            .map(|v| make_elixir_value(env, v))
            .collect::<Vec<_>>()
            .encode(env),
    }
}

impl<'a> Adapter<'a> for AdapterConfig<'a> {
    type Vertex = Arc<StoredTerm>;

    fn resolve_starting_vertices(
        &self,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        _resolve_info: &ResolveInfo,
    ) -> VertexIterator<'static, Self::Vertex> {
        let parameter_data: Vec<(String, Term)> = parameters
            .iter()
            .map(|(k, v)| (k.to_string(), make_elixir_value(self.env, v.to_owned())))
            .collect();

        let rustler_elixir_fun::ElixirFunCallResult::Success(inner) =
            rustler_elixir_fun::apply_elixir_fun(
                self.env,
                self.elixir_invoker,
                self.resolve_starting_vertices,
                (
                    edge_name.as_ref(),
                    rustler::Term::map_from_pairs(self.env, &parameter_data).unwrap(),
                )
                    .encode(self.env),
            )
            .unwrap() else { todo!() };

        let StoredTerm::List(val) = inner else { todo!() };

        Box::new(ElixirVertexIterator::new(val))
    }

    fn resolve_property(
        &self,
        contexts: BaseContextIterator<'static, Self::Vertex>,
        type_name: &Arc<str>,
        property_name: &Arc<str>,
        _resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'static, Self::Vertex, FieldValue> {
        let contexts = encode_contexts(ContextIterator::new(contexts), self.env);
        let rustler_elixir_fun::ElixirFunCallResult::Success(inner) =
            rustler_elixir_fun::apply_elixir_fun(
                self.env,
                self.elixir_invoker,
                self.resolve_property,
                (contexts, type_name.as_ref(), property_name.as_ref()).encode(self.env),
            )
            .unwrap() else { todo!() };

        let StoredTerm::List(val) = inner else { todo!() };

        Box::new(ElixirResolvePropertyIterator::new(val))
    }

    fn resolve_neighbors(
        &self,
        contexts: BaseContextIterator<'static, Self::Vertex>,
        type_name: &Arc<str>,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        _resolve_info: &ResolveEdgeInfo,
    ) -> ContextOutcomeIterator<'static, Self::Vertex, VertexIterator<'static, Self::Vertex>> {
        let contexts = encode_contexts(ContextIterator::new(contexts), self.env);
        let parameter_data: Vec<(String, Term)> = parameters
            .iter()
            .map(|(k, v)| (k.to_string(), make_elixir_value(self.env, v.to_owned())))
            .collect();

        let rustler_elixir_fun::ElixirFunCallResult::Success(inner) =
            rustler_elixir_fun::apply_elixir_fun(
                self.env,
                self.elixir_invoker,
                self.resolve_neighbors,
                (
                    contexts,
                    type_name.as_ref(),
                    edge_name.as_ref(),
                    rustler::Term::map_from_pairs(self.env, &parameter_data).unwrap(),
                )
                    .encode(self.env),
            )
            .unwrap() else { todo!() };

        let StoredTerm::List(val) = inner else { todo!() };

        Box::new(ElixirResolveNeighborsIterator::new(val))
    }

    fn resolve_coercion(
        &self,
        contexts: BaseContextIterator<'static, Self::Vertex>,
        type_name: &Arc<str>,
        coerce_to_type: &Arc<str>,
        _resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'static, Self::Vertex, bool> {
        let contexts = encode_contexts(ContextIterator::new(contexts), self.env);
        let rustler_elixir_fun::ElixirFunCallResult::Success(inner) =
            rustler_elixir_fun::apply_elixir_fun(
                self.env,
                self.elixir_invoker,
                self.resolve_coercion,
                (contexts, type_name.as_ref(), coerce_to_type.as_ref()).encode(self.env),
            )
            .unwrap() else { todo!() };

        let StoredTerm::List(val) = inner else { todo!() };

        Box::new(ElixirResolveCoercionIterator::new(val))
    }
}

fn encode_contexts<'a>(context_iterator: ContextIterator, env: Env<'a>) -> Vec<Term<'a>> {
    let encoded: Vec<Term> = context_iterator
        .0
        .map(|v| v.active_vertex().encode(env))
        .collect();
    encoded
}

fn context_from_stored_term(term: StoredTerm) -> Context {
    Context(DataContext::new(Some(Arc::new(term))))
}

fn stored_term_to_boolean(atom: &Atom) -> bool {
    if atom == &rustler::types::atom::true_() {
        true
    } else {
        false
    }
}

struct ElixirVertexIterator {
    underlying: Vec<StoredTerm>,
}

impl ElixirVertexIterator {
    fn new(underlying: Vec<StoredTerm>) -> Self {
        Self { underlying }
    }
}

impl Iterator for ElixirVertexIterator {
    type Item = Arc<StoredTerm>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.underlying.iter().next() {
            Some(value) => Some(Arc::new(value.clone())),
            None => None,
        }
    }
}

struct ElixirResolvePropertyIterator {
    underlying: Vec<StoredTerm>,
}

impl ElixirResolvePropertyIterator {
    fn new(underlying: Vec<StoredTerm>) -> Self {
        Self { underlying }
    }
}

impl Iterator for ElixirResolvePropertyIterator {
    type Item = (DataContext<Arc<StoredTerm>>, FieldValue);

    fn next(&mut self) -> Option<Self::Item> {
        match self.underlying.iter().next() {
            Some(output) => {
                let StoredTerm::Tuple(output_tuple) = output else { todo!() };

                let context: Context =
                    context_from_stored_term(output_tuple.iter().next().unwrap().clone());
                let value: FieldValue =
                    make_field_value_from_stored_term(output_tuple.iter().next().unwrap()).unwrap();
                Some((context.0, value))
            }
            None => None,
        }
    }
}

struct ElixirResolveNeighborsIterator {
    underlying: Vec<StoredTerm>,
}

impl ElixirResolveNeighborsIterator {
    fn new(underlying: Vec<StoredTerm>) -> Self {
        Self { underlying }
    }
}

impl Iterator for ElixirResolveNeighborsIterator {
    type Item = (
        DataContext<Arc<StoredTerm>>,
        VertexIterator<'static, Arc<StoredTerm>>,
    );

    fn next(&mut self) -> Option<Self::Item> {
        match self.underlying.iter().next() {
            Some(output) => {
                let StoredTerm::Tuple(output_tuple) = output else { todo!() };

                let context: Context =
                    context_from_stored_term(output_tuple.iter().next().unwrap().clone());
                let StoredTerm::List(neighbors_iterable) =
                    output_tuple.iter().next().unwrap().clone() else { todo!() };

                let neighbors: VertexIterator<'static, Arc<StoredTerm>> =
                    Box::new(ElixirVertexIterator::new(neighbors_iterable));
                Some((context.0, neighbors))
            }
            None => None,
        }
    }
}

struct ElixirResolveCoercionIterator {
    underlying: Vec<StoredTerm>,
}

impl ElixirResolveCoercionIterator {
    fn new(underlying: Vec<StoredTerm>) -> Self {
        Self { underlying }
    }
}

impl Iterator for ElixirResolveCoercionIterator {
    type Item = (DataContext<Arc<StoredTerm>>, bool);

    fn next(&mut self) -> Option<Self::Item> {
        match self.underlying.iter().next() {
            Some(output) => {
                let StoredTerm::Tuple(output_tuple) = output else { todo!() };

                let context: Context =
                    context_from_stored_term(output_tuple.iter().next().unwrap().clone());
                let StoredTerm::AnAtom(can_coerce) = output_tuple.iter().next().unwrap() else { todo!() };

                Some((context.0, stored_term_to_boolean(can_coerce)))
            }
            None => None,
        }
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Exposed as NIF for easy testing
/// But normally, you'd want to call `rustler_elixir_fun::apply_elixir_fun`
/// from some other Rust code (rather than from Elixir) instead.
fn apply_elixir_fun<'a>(
    env: Env<'a>,
    pid_or_name: Term<'a>,
    fun: Term<'a>,
    parameters: Term<'a>,
) -> Result<Term<'a>, Error> {
    Ok(rustler_elixir_fun::apply_elixir_fun(env, pid_or_name, fun, parameters)?.encode(env))
}

fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

fn make_field_value_from_term(value: &Term) -> Result<FieldValue, ()> {
    match value.get_type() {
        rustler::TermType::Atom => {
            if let Ok(inner) = value.decode::<rustler::Atom>() {
                if inner == rustler::types::atom::true_() {
                    Ok(FieldValue::Boolean(true))
                } else if inner == rustler::types::atom::false_() {
                    Ok(FieldValue::Boolean(false))
                } else if inner == rustler::types::atom::nil() {
                    Ok(FieldValue::Null)
                } else {
                    Err(())
                }
            } else {
                Err(())
            }
        }
        rustler::TermType::Number =>
        // Do we want to use the if let structure outside of the match?
        {
            if let Ok(inner) = value.decode::<i64>() {
                Ok(FieldValue::Int64(inner))
            } else if let Ok(inner) = value.decode::<f64>() {
                Ok(FieldValue::Float64(inner))
            } else {
                Err(())
            }
        }
        rustler::TermType::Binary => Ok(FieldValue::String(value.decode::<String>().unwrap())),
        rustler::TermType::EmptyList => Ok(FieldValue::List(Vec::new())),
        rustler::TermType::List => {
            if let Ok(inner) = value.decode::<Vec<Term>>() {
                let converted_values = inner.iter().map(make_field_value_from_term).try_fold(
                    vec![],
                    |mut acc, item| {
                        if let Ok(value) = item {
                            acc.push(value);
                            Some(acc)
                        } else {
                            None
                        }
                    },
                );
                if let Some(inner_values) = converted_values {
                    Ok(FieldValue::List(inner_values))
                } else {
                    Err(())
                }
            } else {
                dbg!("failed");
                Err(())
            }
        }
        _ => {
            dbg!(value);
            print!("\r");
            dbg!(value.get_type());
            print!("\r");

            Err(())
        }
    }
}

fn make_field_value_from_stored_term(value: &StoredTerm) -> Result<FieldValue, ()> {
    match value {
        StoredTerm::AnAtom(val) => {
            if val == &rustler::types::atom::true_() {
                Ok(FieldValue::Boolean(true))
            } else if val == &rustler::types::atom::false_() {
                Ok(FieldValue::Boolean(false))
            } else if val == &rustler::types::atom::nil() {
                Ok(FieldValue::Null)
            } else {
                Err(())
            }
        }
        StoredTerm::Integer(val) => Ok(FieldValue::Int64(*val)),
        StoredTerm::Float(val) => Ok(FieldValue::Float64(*val)),
        StoredTerm::Bitstring(val) => Ok(FieldValue::String(val.to_string())),
        StoredTerm::EmptyList() => Ok(FieldValue::List(Vec::new())),

        StoredTerm::List(val) => {
            let converted_values = val.iter().map(&make_field_value_from_stored_term).try_fold(
                vec![],
                |mut acc, item| {
                    if let Ok(value) = item {
                        acc.push(value);
                        Some(acc)
                    } else {
                        None
                    }
                },
            );
            if let Some(inner_values) = converted_values {
                Ok(FieldValue::List(inner_values))
            } else {
                Err(())
            }
        }
        _ => Err(()),
    }
}

// fn to_query_arguments(src: &MapIterator) ->  {
fn to_query_arguments(src: MapIterator) -> Result<Arc<BTreeMap<Arc<str>, FieldValue>>, ()> {
    let mut unrepresentable_args = vec![];
    let mut converted_args = BTreeMap::new();

    for (arg_name, arg_value) in src {
        match make_field_value_from_term(&arg_value) {
            Ok(value) => {
                let inserted =
                    converted_args.insert(Arc::from(arg_name.decode::<String>().unwrap()), value);
                assert!(inserted.is_none());
            }
            Err(_) => {
                unrepresentable_args.push(arg_name);
            }
        }
    }

    if unrepresentable_args.is_empty() {
        Ok(Arc::from(converted_args))
    } else {
        Err(())
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
fn interpret_query<'a>(
    env: Env<'a>,
    adapter: AdapterShim<'a>,
    schema_string: &str,
    query: &str,
    arguments: MapIterator<'a>,
) -> Result<Term<'a>, Error> {
    let my_schema = trustfall_core::schema::Schema::parse(schema_string).expect("failed");

    // if let rustler_elixir_fun::ElixirFunCallResult::Success(inner) =
    //     rustler_elixir_fun::apply_elixir_fun(
    //         env,
    //         adapter.elixir_invoker,
    //         adapter.resolve_starting_vertices,
    //         vec!["hello", "world"].encode(env),
    //     )
    //     .unwrap()
    // {
    //     dbg!(make_field_value_from_stored_term(&inner));
    // }
    let adapter_config = AdapterConfig::new(adapter, env).unwrap();
    let wrapped_adapter = Arc::from(adapter_config);

    let indexed_query = parse(&my_schema, query).unwrap();
    let execution = interpret_ir(
        wrapped_adapter,
        indexed_query,
        to_query_arguments(arguments).unwrap(),
    )
    .unwrap();
    dbg!(query);
    print!("\r");
    print!("{}\n\r", query);

    // println!("{}", type_of(Arc::<str>::from(my_string)));
    // for argument in arguments {
    //     println!("{:?}\r", make_field_value_from_term(&argument.1));
    //     // println!("{:?}\r", argument.1);
    //     // println!("{:?}\r", argument.1.get_type());
    // }
    Ok(vec!["hello", "world"].encode(env))
}

rustler::init!(
    "Elixir.ExTrustfall",
    [add, apply_elixir_fun, interpret_query],
    load = load
);
