use rustler::*;
use rustler_elixir_fun;

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

#[rustler::nif(schedule = "DirtyCpu")]
/// Exposed as NIF for easy testing
/// But normally, you'd want to call `rustler_elixir_fun::apply_elixir_fun`
/// from some other Rust code (rather than from Elixir) instead.
fn apply_elixir_fun<'a>(env: Env<'a>, pid_or_name: Term<'a>, fun: Term<'a>, parameters: Term<'a>) -> Result<Term<'a>, Error> {
    Ok(rustler_elixir_fun::apply_elixir_fun(env, pid_or_name, fun, parameters)?.encode(env))
}

rustler::init!("Elixir.ExTrustfall", [add, apply_elixir_fun], load = load);
