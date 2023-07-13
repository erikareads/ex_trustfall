import Config

if Mix.env() == :dev do
  config :mix_test_watch,
    tasks: [
      "format"
    ]
end
