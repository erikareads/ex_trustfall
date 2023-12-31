defmodule ExTrustfall.MixProject do
  use Mix.Project

  def project do
    [
      app: :ex_trustfall,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:rustler, "~> 0.29.1", override: true},
      {:rustler_elixir_fun, path: "../../rust/elixir-rustler_elixir_fun/"},
      {:mix_test_watch, "~> 1.0", only: :dev, runtime: false}
    ]
  end
end
