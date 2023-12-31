use rustler::error::Error;
use rustler::types::tuple::make_tuple;
use rustler::types::LocalPid;
use rustler::wrapper::ErlNifPid;
use rustler::*;
use rustler_stored_term::term_box::TermBox;
use rustler_stored_term::StoredTerm;
use rustler_stored_term::StoredTerm::{AnAtom, Tuple};
use rustler_sys;
use std::env;
use std::mem::MaybeUninit;
use std::sync::{Condvar, Mutex};
use std::time::Duration;
use wrapper::NIF_ENV;

use crate::ElixirFunCallResult::*;

pub struct ManualFuture {
    mutex: Mutex<Option<StoredTerm>>,
    cond: Condvar,
}

impl ManualFuture {
    pub fn new() -> ManualFuture {
        ManualFuture {
            mutex: Mutex::new(None),
            cond: Condvar::new(),
        }
    }

    pub fn wait_until_filled(&self, timeout: Duration) -> Option<StoredTerm> {
        let (mut guard, wait_timeout_result) = self
            .cond
            .wait_timeout_while(self.mutex.lock().unwrap(), timeout, |pending| {
                pending.is_none()
            })
            .expect("ManualFuture's Mutex was unexpectedly poisoned");
        if wait_timeout_result.timed_out() {
            None
        } else {
            let val = guard.take().unwrap();
            Some(val)
        }
    }
    pub fn fill(&self, value: StoredTerm) {
        let mut started = self.mutex.lock().unwrap();
        *started = Some(value);
        self.cond.notify_all();
    }
}

// pub fn load(env: Env, _info: Term) -> bool {
//     // rustler::resource!(ManualFuture, env);
//     true
// }

mod atoms {
    rustler::atoms! {
        ok,
        error,
        exception,
        exit,
        throw,
        timeout,
    }
}

/// Attempts to turn `name` into a LocalPid
/// Uses [`enif_whereis_pid`](https://www.erlang.org/doc/man/erl_nif.html#enif_whereis_pid) under the hood.
///
/// NOTE: Current implementation is very dirty, as we use transmutation to build a struct whose internals are not exposed by Rustler itself.
/// There is an open PR on Rustler to add support properly: https://github.com/rusterlium/rustler/pull/456
pub fn whereis_pid<'a>(env: Env<'a>, name: Term<'a>) -> Result<LocalPid, Error> {
    let mut enif_pid = MaybeUninit::uninit();
    dbg!("unsafe1");

    if unsafe {
        rustler_sys::enif_whereis_pid(env.as_c_arg(), name.as_c_arg(), enif_pid.as_mut_ptr())
    } == 0
    {
        dbg!("unsafe2");
        Err(Error::Term(Box::new(
            "No pid registered under the given name.",
        )))
    } else {
        dbg!("unsafe4");
        // Safety: Initialized by successful enif call
        let enif_pid = unsafe { enif_pid.assume_init() };
        dbg!("unsafe5");

        // Safety: Safe because `LocalPid` has only one field.
        // NOTE: Dirty hack, but there is no other way to create a LocalPid exposed from `rustler`.
        dbg!("unsafe6");
        let pid = unsafe { std::mem::transmute::<ErlNifPid, LocalPid>(enif_pid) };
        dbg!("unsafe7");
        Ok(pid)
    }
}

fn send_to_elixir<'a>(env: Env<'a>, pid: Term<'a>, value: Term<'a>) -> Result<(), Error> {
    dbg!("sending");
    let pid: LocalPid = pid.decode().or_else(|_| whereis_pid(env, pid))?;
    dbg!("sending2");

    env.send(&pid, value);
    dbg!("sending");
    Ok(())
}
pub fn whereis_pid_owned<'a>(env_c_arg: NIF_ENV, name: Term<'a>) -> Result<LocalPid, Error> {
    let mut enif_pid = MaybeUninit::uninit();
    dbg!("unsafe1");

    if unsafe { rustler_sys::enif_whereis_pid(env_c_arg, name.as_c_arg(), enif_pid.as_mut_ptr()) }
        == 0
    {
        dbg!("unsafe2");
        Err(Error::Term(Box::new(
            "No pid registered under the given name.",
        )))
    } else {
        dbg!("unsafe4");
        // Safety: Initialized by successful enif call
        let enif_pid = unsafe { enif_pid.assume_init() };
        dbg!("unsafe5");

        // Safety: Safe because `LocalPid` has only one field.
        // NOTE: Dirty hack, but there is no other way to create a LocalPid exposed from `rustler`.
        dbg!("unsafe6");
        let pid = unsafe { std::mem::transmute::<ErlNifPid, LocalPid>(enif_pid) };
        dbg!("unsafe7");
        Ok(pid)
    }
}

fn send_to_elixir_owned<'a>(
    pid: rustler::LocalPid,
    name: Term<'_>,
    value: TermBox,
    env_c_arg: NIF_ENV,
) -> Result<(), Error> {
    let pid: LocalPid = name
        .decode()
        .or_else(|_| whereis_pid_owned(env_c_arg, name))?;
    std::thread::spawn(move || {
        OwnedEnv::new().send_and_clear(&pid, |env| value.get(env));
    });
    Ok(())
}

#[derive(Clone)]
/// The result of calling a function on the Elixir side.
///
/// This enum exists because we want to handle all possible failure scenarios correctly.
///
/// ElixirFunCallResult implements the `rustler::types::Encoder` trait,
/// to allow you to convert the result back into a `Term<'a>` representation for easy debugging.
///
/// However, more useful is usually to pattern-match in Rust on the resulting values instead,
/// and only encode the inner `StoredTerm` afterwards.
pub enum ElixirFunCallResult {
    /// The happy path: The function completed successfully. In Elixir, this looks like `{:ok, value}`
    Success(StoredTerm),
    /// An exception was raised. In Elixir, this looks like `{:error, {:exception, some_exception}}`
    ExceptionRaised(StoredTerm),
    /// The code attempted to exit the process using a call to `exit(val)`. In Elixir, this looks like `{:error, {:exit, val}}`
    Exited(StoredTerm),
    /// A raw value was thrown using `throw(val)`. In Elixir, this looks like `{:error, {:thrown, val}}`
    ValueThrown(StoredTerm),
    /// The function took too long to complete. In Elixir, this looks like `{:error, :timeout}`
    TimedOut,
}

impl Encoder for ElixirFunCallResult {
    fn encode<'a>(&self, env: Env<'a>) -> Term<'a> {
        let result = match self {
            Success(term) => Ok(term),
            ExceptionRaised(term) => Err(make_tuple(
                env,
                &[atoms::exception().to_term(env), term.encode(env)],
            )),
            Exited(term) => Err(make_tuple(
                env,
                &[atoms::exit().to_term(env), term.encode(env)],
            )),
            ValueThrown(term) => Err(make_tuple(
                env,
                &[atoms::throw().to_term(env), term.encode(env)],
            )),
            TimedOut => Err(atoms::timeout().to_term(env)),
        };

        result.encode(env)
    }
}

/// Will run `fun` with the parameters `parameters`
/// on the process indicated by `pid_or_name`.
///
/// 'Raises' an ArgumentError (e.g. returns `Err(Error::BadArg)` on the Rust side) if `fun` is not a function or `parameters` is not a list.
///
/// Even with proper parameters, the function call itself might fail.
/// All various scenarios are handled by the `ElixirFunCallResult` type.
///
/// # Notes
///
/// - It waits for a maximum of 5000 milliseconds before returning an `Ok(TimedOut)`.
/// - Be sure to register any NIF that calls this function as a 'Dirty CPU NIF'! (by using `#[rustler::nif(schedule = "DirtyCpu")]`).
///   This is important for two reasons:
///     1. calling back into Elixir might indeed take quite some time.
///     2. we want to prevent schedulers to wait for themselves, which might otherwise sometimes happen.
pub fn apply_elixir_fun<'a>(
    env: Env<'a>,
    pid_or_name: Term<'a>,
    fun: Term<'a>,
    parameters: Term<'a>,
) -> Result<ElixirFunCallResult, Error> {
    apply_elixir_fun_timeout(
        env,
        pid_or_name,
        fun,
        parameters,
        Duration::from_millis(5000),
    )
}

/// Works the same as `apply_elixir_fun` but allows customizing the timeout to wait for the function to return.
pub fn apply_elixir_fun_timeout<'a>(
    env: Env<'a>,
    pid_or_name: Term<'a>,
    fun: Term<'a>,
    parameters: Term<'a>,
    timeout: Duration,
) -> Result<ElixirFunCallResult, Error> {
    if !fun.is_fun() {
        return Err(Error::BadArg);
    }

    if !parameters.is_list() {
        return Err(Error::BadArg);
    }

    // let future = ResourceArc::new(ManualFuture::new());
    // let fun_tuple = rustler::types::tuple::make_tuple(env, &[fun, parameters, future.encode(env)]);
    let future = ManualFuture::new();
    let future_ptr: *const ManualFuture = &future;
    let raw_future_ptr = future_ptr as usize;
    let fun_tuple =
        rustler::types::tuple::make_tuple(env, &[fun, parameters, raw_future_ptr.encode(env)]);
    send_to_elixir(env, pid_or_name, fun_tuple)?;

    match future.wait_until_filled(timeout) {
        None => Ok(TimedOut),
        Some(result) => Ok(parse_fun_call_result(env, result)),
    }
}
pub fn apply_elixir_fun_timeout_owned<'a>(
    env_c_arg: NIF_ENV,
    pid: rustler::LocalPid,
    env: Env<'a>,
    pid_or_name: Term<'a>,
    fun: Term<'a>,
    parameters: Term<'a>,
    timeout: Duration,
) -> Result<ElixirFunCallResult, Error> {
    if !fun.is_fun() {
        return Err(Error::BadArg);
    }

    if !parameters.is_list() {
        return Err(Error::BadArg);
    }

    // let future = ResourceArc::new(ManualFuture::new());
    // let fun_tuple = rustler::types::tuple::make_tuple(env, &[fun, parameters, future.encode(env)]);
    let future = ManualFuture::new();
    let future_ptr: *const ManualFuture = &future;
    let raw_future_ptr = future_ptr as usize;
    let fun_tuple = <rustler_stored_term::term_box::TermBox>::new(
        (&rustler::types::tuple::make_tuple(env, &[fun, parameters, raw_future_ptr.encode(env)])),
    );
    send_to_elixir_owned(pid, pid_or_name, fun_tuple, env_c_arg)?;

    match future.wait_until_filled(timeout) {
        None => Ok(TimedOut),
        Some(result) => Ok(parse_fun_call_result(env, result)),
    }
}

fn parse_fun_call_result<'a>(env: Env<'a>, result: StoredTerm) -> ElixirFunCallResult {
    match result {
        Tuple(ref tuple) =>
            match &tuple[..] {
                [AnAtom(ok), value] if ok == &rustler::types::atom::ok() => Success(value.clone()),
                [AnAtom(error), Tuple(ref error_tuple)] if error == &atoms::error() => {
                    match &error_tuple[..] {
                        [AnAtom(exception), problem] if exception == &atoms::exception() => ExceptionRaised(problem.clone()),
                        [AnAtom(exit), problem] if exit == &atoms::exit() => Exited(problem.clone()),
                        [AnAtom(throw), problem] if throw == &atoms::throw() => ValueThrown(problem.clone()),
                        _ => panic!("RustlerElixirFun's function wrapper returned an unexpected error tuple result: {:?}", result.encode(env))
                    }
                },
                _ => panic!("RustlerElixirFun's function wrapper returned an unexpected tuple result: {:?}", result.encode(env))
            },
        _ => panic!("RustlerElixirFun's function wrapper returned an unexpected result: {:?}", result.encode(env))
    }
}
