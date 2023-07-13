defmodule ExTrustfall.Context do
  defstruct [:active_vertex]

  def active_vertex(context), do: context.active_vertex
end

defmodule ExTrustfall.Adapter do
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
              edge_name: String.t(),
              parameters: %{optional(String.t()) => any()}
            ) :: [vertex()]

  @callback resolve_property(
              contexts: [%Context{active_vertex: vertex()}],
              type_name: String.t(),
              property_name: String.t()
            ) :: [{%Context{active_vertex: vertex()}, term()}]

  @callback resolve_neighbors(
              contexts: [%Context{active_vertex: vertex()}],
              type_name: String.t(),
              edge_name: String.t(),
              parameters: %{optional(String.t()) => any()}
            ) :: [{%Context{active_vertex: vertex()}, [vertex()]}]

  @callback resolve_coercion(
              contexts: [%Context{active_vertex: vertex()}],
              type_name: String.t(),
              coerce_to_type: String.t()
            ) :: [{%Context{active_vertex: vertex()}, boolean()}]
end

defmodule ExTrustfall do
  use Rustler,
    otp_app: :ex_trustfall,
    crate: :ex_trustfall

  def add(_x, _y), do: :erlang.nif_error(:nif_not_loaded)
  def apply_elixir_fun(_pid_or_name, _fun, _params), do: :erlang.nif_error(:nif_not_loaded)
end
