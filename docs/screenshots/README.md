# Operator UI screenshots

`before/` contains the prior interface and `after/` contains the redesign. Each directory has screenshots for idle, running, awaiting approval, verified, needs-input, and failed states at 1440, 1024, and 800 CSS pixels wide.

The state captures use Playwright with synthetic responses shaped like the existing API payloads. They exercise the UI without running or changing the invoice agent. The post-change capture also checked for horizontal overflow, visible text below 14px, browser errors, the approval state, verification evidence, and the raw diagnostics event count.
