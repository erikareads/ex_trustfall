defmodule ExTrustfall.Context do
  defstruct [:active_vertex]

  def active_vertex(context), do: context.active_vertex
end

defmodule ExTrustfall.AdapterShim do
  defstruct [
    :elixir_invoker,
    :resolve_starting_vertices,
    :resolve_property,
    :resolve_neighbors,
    :resolve_coercion
  ]

  alias ExTrustfall.Context

  def create(implementing_module, pid_or_name) do
    %__MODULE__{
      elixir_invoker: pid_or_name,
      resolve_starting_vertices: &implementing_module.resolve_starting_vertices/2,
      resolve_property: &implementing_module.resolve_property/3,
      resolve_neighbors: &implementing_module.resolve_neighbors/4,
      resolve_coercion: &implementing_module.resolve_coercion/3
    }
  end

  @type vertex :: term()
  @callback resolve_starting_vertices(
              edge_name :: String.t(),
              parameters :: %{optional(String.t()) => any()}
            ) :: [vertex()]

  @callback resolve_property(
              contexts :: [%Context{active_vertex: vertex()}],
              type_name :: String.t(),
              property_name :: String.t()
            ) :: [{%Context{active_vertex: vertex()}, term()}]

  @callback resolve_neighbors(
              contexts :: [%Context{active_vertex: vertex()}],
              type_name :: String.t(),
              edge_name :: String.t(),
              parameters :: %{optional(String.t()) => any()}
            ) :: [{%Context{active_vertex: vertex()}, [vertex()]}]

  @callback resolve_coercion(
              contexts :: [%Context{active_vertex: vertex()}],
              type_name :: String.t(),
              coerce_to_type :: String.t()
            ) :: [{%Context{active_vertex: vertex()}, boolean()}]
end

defmodule ExTrustfall do
  use Rustler,
    otp_app: :ex_trustfall,
    crate: :ex_trustfall

  def add(_x, _y), do: :erlang.nif_error(:nif_not_loaded)
  def apply_elixir_fun(_pid_or_name, _fun, _params), do: :erlang.nif_error(:nif_not_loaded)

  def execute_query(adapter, schema, query, arguments) do
    interpret_query(adapter, schema, query, arguments)
  end

  defp interpret_query(_adapter, _schema, _query, _arguments),
    do: :erlang.nif_error(:nif_not_loaded)
end

defmodule ExTrustfall.Example do
  @behaviour ExTrustfall.AdapterShim
  @number_names [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten"
  ]

  @impl true
  def resolve_starting_vertices(edge_name, parameters) do
    IO.inspect({edge_name, parameters})
    max_value = parameters["max"]
    0..max_value |> Enum.to_list()
  end

  @impl true
  def resolve_property(contexts, type_name, property_name) do
    IO.inspect({contexts, type_name, property_name})

    Enum.map(contexts, fn context ->
      vertex = context.active_vertex

      value =
        cond do
          property_name == "value" -> vertex
          property_name == "name" -> @number_names |> Enum.at(vertex)
        end

      {context, value}
    end)
  end

  @impl true
  def resolve_neighbors(contexts, type_name, edge_name, parameters) do
    IO.inspect({contexts, type_name, edge_name, parameters})

    Enum.map(contexts, fn context ->
      vertex = context.active_vertex

      neighbors =
        cond do
          edge_name == "multiple" and vertex > 0 ->
            max_value = parameters["max"]
            start = 2 * vertex
            ending = max_value * vertex + 1
            step = vertex
            start..ending//step |> Enum.to_list()

          edge_name == "predecessor" and vertex > 0 ->
            [vertex - 1]

          edge_name == "successor" ->
            [vertex + 1]
        end

      {context, neighbors}
    end)
  end

  @impl true
  def resolve_coercion(_contexts, _type_name, _coerce_to_type) do
    []
  end
end
