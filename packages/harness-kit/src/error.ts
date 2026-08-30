export class HarnessContractError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "HarnessContractError";
    this.code = code;
  }
}
