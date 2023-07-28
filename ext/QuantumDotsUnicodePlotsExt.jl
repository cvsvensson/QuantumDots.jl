
module QuantumDotsUnicodePlotsExt

using QuantumDots, UnicodePlots
import QuantumDots: QuasiParticle, labels, basis, visualize, majvisualize

function majvisualize(qp::QuasiParticle)
    ls = labels(basis(qp))
    for (γ, title) in (((qp + qp') / 2, "γ₊ = (χ + χ')/2"), ((qp - qp') / 2, "γ₋ = (χ - χ')/2"))
        xlabels = map(l -> (l, :x), ls)
        ylabels = map(l -> (l, :y), ls)
        xweights = map(l -> γ[l, :h] + γ[l, :p], ls)
        yweights = map(l -> γ[l, :h] - γ[l, :p], ls)
        display(barplot(xlabels, abs2.(xweights); title, maximum=1, border=:ascii))
        display(barplot(ylabels, abs2.(yweights), maximum=1, border=:dashed))
    end
end
function visualize(qp::QuasiParticle)
    hlabels = map(l -> (l, :h), labels(qp.basis))
    plabels = map(l -> (l, :p), labels(qp.basis))
    hweights = map(l -> qp[l], hlabels)
    pweights = map(l -> qp[l], plabels)
    display(barplot(hlabels, abs2.(hweights), title="Quasiparticle", maximum=1, border=:ascii))
    display(barplot(plabels, abs2.(pweights), maximum=1, border=:dashed))
end

end