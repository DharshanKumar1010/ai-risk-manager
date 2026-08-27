/** Minimal linear scale + "nice" tick generation. No d3-scale dependency: a linear map and a
 * tick step are a handful of lines, and every chart in this panel needs the same one. */

export interface LinearScale {
  (value: number): number
  domain: [number, number]
  range: [number, number]
}

export function linearScale(domain: [number, number], range: [number, number]): LinearScale {
  const [d0, d1] = domain
  const [r0, r1] = range
  const span = d1 - d0
  const scale = ((value: number) => {
    if (span === 0) return r0
    return r0 + ((value - d0) / span) * (r1 - r0)
  }) as LinearScale
  scale.domain = domain
  scale.range = range
  return scale
}

/** "Nice" round tick values spanning `domain`, roughly `count` of them -- the same D3-style
 * algorithm every charting library implements: round the step to 1/2/5 * 10^n. */
export function niceTicks(domain: [number, number], count = 5): number[] {
  const [lo, hi] = domain
  if (lo === hi) return [lo]
  const span = hi - lo
  const rawStep = span / count
  const magnitude = 10 ** Math.floor(Math.log10(rawStep))
  const residual = rawStep / magnitude
  const niceResidual = residual >= 5 ? 10 : residual >= 2 ? 5 : residual >= 1 ? 2 : 1
  const step = niceResidual * magnitude

  const start = Math.ceil(lo / step) * step
  const ticks: number[] = []
  for (let value = start; value <= hi + step * 1e-9; value += step) {
    ticks.push(Number(value.toFixed(10)))
  }
  return ticks
}
