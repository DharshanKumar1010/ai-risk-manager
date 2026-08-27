/** Join truthy class-name fragments. No variant merging -- see components/ui's README note on
 * why tailwind-merge is not pulled in for a set of primitives with fixed class strings. */
export function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(' ')
}
