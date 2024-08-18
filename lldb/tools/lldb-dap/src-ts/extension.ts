import * as vscode from "vscode";
import * as fs from "node:fs/promises";
import { LLDBDapOptions } from "./types";
import { DisposableContext } from "./disposable-context";
import { LLDBDapDescriptorFactory } from "./debug-adapter-factory";

/**
 * This creates the configurations for this project if used as a standalone
 * extension.
 */
function createDefaultLLDBDapOptions(): LLDBDapOptions {
  return {
    debuggerType: "lldb-dap",
    async createDapExecutableCommand(
      session: vscode.DebugSession,
      packageJSONExecutable: vscode.DebugAdapterExecutable | undefined,
    ): Promise<vscode.DebugAdapterExecutable | undefined> {
      const path = vscode.workspace
        .getConfiguration("lldb-dap", session.workspaceFolder)
        .get<string>("executable-path");

      if (!path) {
        return packageJSONExecutable;
      }

      try {
        const fileStats = await fs.stat(path);
        if (!fileStats.isFile()) {
          throw new Error(`Error: ${path} is not a file`);
        }
      } catch (err) {
        const error: Error = err as Error;
        const openSettingsAction = "Open Settings";
        const callBackValue = await vscode.window.showErrorMessage(
          error.message,
          { modal: true },
          openSettingsAction,
        );
        if (openSettingsAction === callBackValue) {
          vscode.commands.executeCommand(
            "workbench.action.openSettings",
            "lldb-dap.executable-path",
          );
        }
      }
      return new vscode.DebugAdapterExecutable(path, []);
    },
  };
}

/**
 * This class represents the extension and manages its life cycle. Other extensions
 * using it as as library should use this class as the main entry point.
 */
export class LLDBDapExtension extends DisposableContext {
  private lldbDapOptions: LLDBDapOptions;

  constructor(lldbDapOptions: LLDBDapOptions) {
    super();
    this.lldbDapOptions = lldbDapOptions;

    this.pushSubscription(
      vscode.debug.registerDebugAdapterDescriptorFactory(
        this.lldbDapOptions.debuggerType,
        new LLDBDapDescriptorFactory(this.lldbDapOptions),
      ),
    );
  }
}

/**
 * This is the entry point when initialized by VS Code.
 */
export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(
    new LLDBDapExtension(createDefaultLLDBDapOptions()),
  );
}
