using QuantumDots
using Documenter

DocMeta.setdocmeta!(QuantumDots, :DocTestSetup, :(using QuantumDots); recursive=true)

makedocs(;
    modules=[QuantumDots],
    authors="Viktor Svensson and William Samuelson", sitename="QuantumDots.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cvsvensson.github.io/QuantumDots.jl",
        edit_link="main",
        assets=String[],
        repolink="https://github.com/cvsvensson/QuantumDots.jl/blob/{commit}{path}#{line}",
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cvsvensson/QuantumDots.jl",
    devbranch="main",
)
