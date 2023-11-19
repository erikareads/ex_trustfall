use rustler::env::SavedTerm;
use rustler::*;
use rustler_elixir_fun;
use rustler_stored_term::StoredTerm;
use std::any::type_name;
use std::fmt::Debug;
use std::time::Duration;
use std::{collections::BTreeMap, sync::Arc};
use trustfall_core::{
    frontend::parse,
    interpreter::{
        execution::interpret_ir, Adapter, AsVertex, ContextIterator as BaseContextIterator,
        ContextOutcomeIterator, DataContext, ResolveEdgeInfo, ResolveInfo, VertexIterator,
    },
    ir::{EdgeParameters, FieldValue},
};
use wrapper::NIF_ENV;

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

#[derive(NifStruct, Debug, Clone)]
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

pub(crate) struct Opaque {
    data: *mut (),
    pub(crate) vertex: Option<Arc<StoredTerm>>,
}

impl Opaque {
    fn new<V: AsVertex<Arc<StoredTerm>> + 'static>(ctx: DataContext<V>) -> Self {
        let vertex = ctx.active_vertex::<Arc<StoredTerm>>().cloned();
        let boxed = Box::new(ctx);
        let data = Box::into_raw(boxed) as *mut ();

        Self { data, vertex }
    }

    /// Converts an `Opaque` into the `DataContext<V>` it points to.
    ///
    /// # Safety
    ///
    /// When an `Opaque` is constructed, it does not store the value of the `V` generic parameter
    /// it was constructed with. The caller of this function must ensure that the `V` parameter here
    /// is the same type as the one used in the `Opaque::new()` call that constructed `self` here.
    unsafe fn into_inner<V: AsVertex<Arc<StoredTerm>> + 'static>(self) -> DataContext<V> {
        let boxed_ctx = unsafe { Box::from_raw(self.data as *mut DataContext<V>) };
        *boxed_ctx
    }
}

impl Opaque {
    // fn active_vertex(&self) -> PyResult<Option<Py<PyAny>>> {
    //     Ok(self.vertex.as_ref().map(|arc| (**arc).clone()))
    // }
    fn active_vertex(&self) -> Result<Option<StoredTerm>, ()> {
        Ok(self.vertex.as_ref().map(|arc| (**arc).clone()))
    }
}

pub struct ContextIterator(VertexIterator<'static, Opaque>);

impl ContextIterator {
    fn new<V: AsVertex<Arc<StoredTerm>> + 'static>(inner: BaseContextIterator<'static, V>) -> Self {
        Self(Box::new(inner.map(Opaque::new)))
    }
}

struct AdapterConfig {
    env_c_arg: NIF_ENV,
    pid: rustler::LocalPid,
    elixir_invoker: SavedTerm,
    resolve_starting_vertices: SavedTerm,
    resolve_property: SavedTerm,
    resolve_neighbors: SavedTerm,
    resolve_coercion: SavedTerm,
    env: OwnedEnv,
}

impl AdapterConfig {
    pub fn new<'a>(
        adapter_shim: AdapterShim<'a>,
        owned_env: OwnedEnv,
        pid: rustler::LocalPid,
        env_c_arg: NIF_ENV,
    ) -> Result<Self, ()> {
        let elixir_invoker = owned_env.save(adapter_shim.elixir_invoker);
        let resolve_starting_vertices = owned_env.save(adapter_shim.resolve_starting_vertices);
        let resolve_property = owned_env.save(adapter_shim.resolve_property);
        let resolve_neighbors = owned_env.save(adapter_shim.resolve_neighbors);
        let resolve_coercion = owned_env.save(adapter_shim.resolve_coercion);
        let env = owned_env;

        Ok(Self {
            elixir_invoker,
            resolve_starting_vertices,
            resolve_property,
            resolve_neighbors,
            resolve_coercion,
            env,
            pid,
            env_c_arg,
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
        FieldValue::Enum(_) => todo!(),
        FieldValue::List(x) => x
            .into_iter()
            .map(|v| make_elixir_value(env, v.clone()))
            .collect::<Vec<_>>()
            .encode(env),
        _ => todo!(),
    }
}

fn make_stored_term(value: FieldValue) -> StoredTerm {
    match value {
        FieldValue::Null => StoredTerm::AnAtom(rustler::types::atom::nil()),
        FieldValue::Uint64(x) => StoredTerm::Integer(x.try_into().unwrap()),
        FieldValue::Int64(x) => StoredTerm::Integer(x),
        FieldValue::Float64(x) => StoredTerm::Float(x),
        FieldValue::String(x) => StoredTerm::Bitstring(x.as_ref().to_string()),
        FieldValue::Boolean(true) => StoredTerm::AnAtom(rustler::types::atom::true_()),
        FieldValue::Boolean(false) => StoredTerm::AnAtom(rustler::types::atom::false_()),
        FieldValue::Enum(_) => todo!(),
        FieldValue::List(x) => StoredTerm::List(
            x.into_iter()
                .map(|v| make_stored_term(v.clone()))
                .collect::<Vec<_>>(),
        ),
        _ => todo!(),
    }
}

impl Adapter<'static> for AdapterConfig {
    type Vertex = Arc<StoredTerm>;

    fn resolve_starting_vertices(
        &self,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        _resolve_info: &ResolveInfo,
    ) -> VertexIterator<'static, Self::Vertex> {
        dbg!("hello6");

        let parameter_data: Vec<(String, StoredTerm)> = {
            parameters
                .iter()
                .map(|(k, v)| (k.to_string(), make_stored_term(v.to_owned())))
                .collect()
        };
        dbg!("hello7");

        let rustler_elixir_fun::ElixirFunCallResult::Success(inner) = self.env.run(|env| {
            dbg!("before");
            dbg!(env.as_c_arg());
            let result = rustler_elixir_fun::apply_elixir_fun_timeout_owned(
                self.env_c_arg,
                self.pid,
                env,
                self.elixir_invoker.load(env),
                self.resolve_starting_vertices.load(env),
                vec![
                    edge_name.as_ref().encode(env),
                    rustler::Term::map_from_pairs(
                        env,
                        &parameter_data
                            .iter()
                            .map(|(k, v)| (k, v.encode(env)))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap()
                    .encode(env),
                ]
                .encode(env),
                Duration::from_millis(500),
            );
            dbg!("here");
            result.unwrap()
        }) else {
            dbg!("blow up");
            todo!()
        };
        dbg!("hello8");

        let StoredTerm::List(val) = inner else {
            todo!()
        };

        let vertex_iterator = ElixirVertexIterator::new(val);
        dbg!(vertex_iterator.clone());
        Box::new(vertex_iterator)
    }

    fn resolve_property<V: AsVertex<Self::Vertex> + 'static>(
        &self,
        contexts: BaseContextIterator<'static, V>,
        type_name: &Arc<str>,
        property_name: &Arc<str>,
        _resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'static, V, FieldValue> {
        let contexts = ContextIterator::new(contexts);
        let rustler_elixir_fun::ElixirFunCallResult::Success(inner) = self.env.run(|env| {
            rustler_elixir_fun::apply_elixir_fun_timeout_owned(
                self.env_c_arg,
                self.pid,
                env,
                self.resolve_property.load(env),
                self.elixir_invoker.load(env),
                vec![
                    contexts
                        .0
                        .map(|v| v.active_vertex().encode(env))
                        .collect::<Vec<Term>>()
                        .encode(env),
                    type_name.as_ref().encode(env),
                    property_name.as_ref().encode(env),
                ]
                .encode(env),
                Duration::from_millis(500),
            )
            .unwrap()
        }) else {
            todo!()
        };

        let StoredTerm::List(val) = inner else {
            todo!()
        };

        let iter = ElixirResolvePropertyIterator::new(val);

        //Box::new(ElixirResolvePropertyIterator::new(val))
        Box::new(iter.map(|(opaque, value)| {
            // SAFETY: This `Opaque` was constructed just a few lines ago
            //         in this `resolve_property()` call, so the `V` type must be the same.
            let ctx = unsafe { opaque.into_inner() };

            (ctx, value)
        }))
    }

    fn resolve_neighbors<V: AsVertex<Self::Vertex> + 'static>(
        &self,
        contexts: BaseContextIterator<'static, V>,
        type_name: &Arc<str>,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        _resolve_info: &ResolveEdgeInfo,
    ) -> ContextOutcomeIterator<'static, V, VertexIterator<'static, Self::Vertex>> {
        let contexts = ContextIterator::new(contexts);
        let parameter_data: Vec<(String, StoredTerm)> = {
            parameters
                .iter()
                .map(|(k, v)| (k.to_string(), make_stored_term(v.to_owned())))
                .collect()
        };

        let rustler_elixir_fun::ElixirFunCallResult::Success(inner) = self.env.run(|env| {
            rustler_elixir_fun::apply_elixir_fun_timeout_owned(
                self.env_c_arg,
                self.pid,
                env,
                self.resolve_neighbors.load(env),
                self.elixir_invoker.load(env),
                vec![
                    contexts
                        .0
                        .map(|v| v.active_vertex().encode(env))
                        .collect::<Vec<Term>>()
                        .encode(env),
                    type_name.as_ref().encode(env),
                    edge_name.as_ref().encode(env),
                    rustler::Term::map_from_pairs(
                        env,
                        &parameter_data
                            .iter()
                            .map(|(k, v)| (k, v.encode(env)))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap()
                    .encode(env),
                ]
                .encode(env),
                Duration::from_millis(500),
            )
            .unwrap()
        }) else {
            todo!()
        };

        let StoredTerm::List(val) = inner else {
            todo!()
        };
        let iter = ElixirResolveNeighborsIterator::new(val);
        Box::new(iter.map(|(opaque, neighbors)| {
            // SAFETY: This `Opaque` was constructed just a few lines ago
            //         in this `resolve_neighbors()` call, so the `V` type must be the same.
            let ctx = unsafe { opaque.into_inner() };

            (ctx, neighbors)
        }))
    }

    fn resolve_coercion<V: AsVertex<Self::Vertex> + 'static>(
        &self,
        contexts: BaseContextIterator<'static, V>,
        type_name: &Arc<str>,
        coerce_to_type: &Arc<str>,
        _resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'static, V, bool> {
        let contexts = ContextIterator::new(contexts);
        let rustler_elixir_fun::ElixirFunCallResult::Success(inner) = self.env.run(|env| {
            rustler_elixir_fun::apply_elixir_fun_timeout_owned(
                self.env_c_arg,
                self.pid,
                env,
                self.resolve_coercion.load(env),
                self.elixir_invoker.load(env),
                vec![
                    contexts
                        .0
                        .map(|v| v.active_vertex().encode(env))
                        .collect::<Vec<Term>>()
                        .encode(env),
                    type_name.as_ref().encode(env),
                    coerce_to_type.as_ref().encode(env),
                ]
                .encode(env),
                Duration::from_millis(500),
            )
            .unwrap()
        }) else {
            todo!()
        };

        let StoredTerm::List(val) = inner else {
            todo!()
        };

        let iter = ElixirResolveCoercionIterator::new(val);
        Box::new(iter.map(|(opaque, value)| {
            // SAFETY: This `Opaque` was constructed just a few lines ago
            //         in this `resolve_coercion()` call, so the `V` type must be the same.
            let ctx = unsafe { opaque.into_inner() };

            (ctx, value)
        }))
    }
}

// fn encode_contexts(context_iterator: ContextIterator, env: OwnedEnv) -> Vec<SavedTerm> {
//     let encoded: Vec<SavedTerm> = context_iterator
//         .0
//         .map(|v| env.save(env.run(|sub_env| v.active_vertex().encode(sub_env))))
//         .collect();
//     encoded
// }

fn context_from_stored_term(term: StoredTerm) -> Opaque {
    //Context(DataContext::new(Some(Arc::new(term))))
    Opaque::new(DataContext::new(Some(Arc::new(term))))
}

fn stored_term_to_boolean(atom: &Atom) -> bool {
    if atom == &rustler::types::atom::true_() {
        true
    } else {
        false
    }
}

#[derive(Debug, Clone)]
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
    type Item = (Opaque, FieldValue);

    fn next(&mut self) -> Option<Self::Item> {
        match self.underlying.iter().next() {
            Some(output) => {
                let StoredTerm::Tuple(output_tuple) = output else {
                    todo!()
                };

                let context: Opaque =
                    context_from_stored_term(output_tuple.iter().next().unwrap().clone());
                let value: FieldValue =
                    make_field_value_from_stored_term(output_tuple.iter().next().unwrap()).unwrap();
                Some((context, value))
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
    type Item = (Opaque, VertexIterator<'static, Arc<StoredTerm>>);

    fn next(&mut self) -> Option<Self::Item> {
        match self.underlying.iter().next() {
            Some(output) => {
                let StoredTerm::Tuple(output_tuple) = output else {
                    todo!()
                };

                let context = context_from_stored_term(output_tuple.iter().next().unwrap().clone());
                let StoredTerm::List(neighbors_iterable) =
                    output_tuple.iter().next().unwrap().clone()
                else {
                    todo!()
                };

                let neighbors: VertexIterator<'static, Arc<StoredTerm>> =
                    Box::new(ElixirVertexIterator::new(neighbors_iterable));
                Some((context, neighbors))
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
    type Item = (Opaque, bool);

    fn next(&mut self) -> Option<Self::Item> {
        match self.underlying.iter().next() {
            Some(output) => {
                let StoredTerm::Tuple(output_tuple) = output else {
                    todo!()
                };

                let context: Opaque =
                    context_from_stored_term(output_tuple.iter().next().unwrap().clone());
                let StoredTerm::AnAtom(can_coerce) = output_tuple.iter().next().unwrap() else {
                    todo!()
                };

                Some((context, stored_term_to_boolean(can_coerce)))
            }
            None => None,
        }
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Exposed as NIF for easy testing
/// But normally, you'd want to call `rustler_elixir_fun::apply_elixir_fun_timeout_owned`
/// from some other Rust code (rather than from Elixir) instead.
fn apply_elixir_fun<'a>(
    env: Env<'a>,
    pid_or_name: Term<'a>,
    fun: Term<'a>,
    _parameters: Term<'a>,
) -> Result<Term<'a>, Error> {
    Ok(rustler_elixir_fun::apply_elixir_fun(
        env,
        pid_or_name,
        fun,
        vec!["hello".encode(env), 1.encode(env)].encode(env),
    )?
    .encode(env))
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
        rustler::TermType::Binary => {
            Ok(FieldValue::String(value.decode::<String>().unwrap().into()))
        }
        rustler::TermType::EmptyList => Ok(FieldValue::List(Vec::new().into())),
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
                    Ok(FieldValue::List(inner_values.into()))
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
        StoredTerm::Bitstring(val) => Ok(FieldValue::String(val.to_string().into())),
        StoredTerm::EmptyList() => Ok(FieldValue::List(Vec::new().into())),

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
                Ok(FieldValue::List(inner_values.into()))
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
    //     rustler_elixir_fun::apply_elixir_fun_timeout_owned(
    //         env,
    //         adapter.elixir_invoker,
    //         adapter.resolve_starting_vertices,
    //         vec!["hello", "world"].encode(env),
    //     )
    //     .unwrap()
    // {
    //     dbg!(make_field_value_from_stored_term(&inner));
    // }
    let owned_env = rustler::OwnedEnv::new();
    dbg!(adapter.clone());

    let adapter_config = AdapterConfig::new(adapter, owned_env, env.pid(), env.as_c_arg()).unwrap();

    dbg!("hello2");
    let wrapped_adapter = Arc::from(adapter_config);
    dbg!("hello3");

    let indexed_query = parse(&my_schema, query).unwrap();
    dbg!("hello4");
    let execution = interpret_ir(
        wrapped_adapter,
        indexed_query,
        to_query_arguments(arguments).unwrap(),
    )
    .unwrap();
    dbg!("dead");
    let owned_iter: Box<dyn Iterator<Item = BTreeMap<String, Term>>> =
        Box::new(execution.map(|res| {
            res.into_iter()
                .map(|(k, v)| {
                    let elixir_value = make_elixir_value(env, v);
                    (k.to_string(), elixir_value)
                })
                .collect()
        }));
    for item in owned_iter {
        dbg!(item);
    }

    //Ok(ResultIterator { iter: owned_iter });
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
