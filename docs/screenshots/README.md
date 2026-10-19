# Operator UI screenshots

`before/` contains the prior interface and `after/` contains the redesign. The redesign has screenshots for idle, running, awaiting approval, verified, needs-input, and failed states at 1440, 1024, 800, and 390 CSS pixels wide.

Those 24 responsive state captures use Playwright with synthetic responses shaped like the existing API payloads. They exercise the UI without running or changing the invoice agent. The `flagship-*.png` captures show a separate real offline replay through the UI: Acme latest-invoice selection, approval, ERP entry, and six-field verification. `flagship-evidence-1440.png` is a close view of the six-field match table and ledger screenshot.

The responsive capture pass checked for horizontal overflow, visible text below 14px, and browser console errors at every state and width. It also checked the approval card, verification evidence, ledger screenshot preview, and diagnostics event rendering. The real offline replay checked the approval facts and policy reason, then confirmed that all six evidence fields matched after submission.
