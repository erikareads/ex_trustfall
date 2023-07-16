{:ok, _} = RustlerElixirFun.FunExecutionServer.start_link(name: Server)
adapter = ExTrustfall.AdapterShim.create(ExTrustfall.Example, Server)
schema = File.read!("schema.graphql")
query = """
{
    Number(max: 4) {
        value @output
        
        multiple(max: 3) {
            mul: value @output
        }
    }
}
"""

IO.inspect(ExTrustfall.execute_query(adapter, schema, query, %{}))
