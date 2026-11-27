<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <meta name="theme-color" content="#38BDF8" />
  <title>RespiKidd App</title>
  <link rel="manifest" href="manifest.json" />
  <style>
    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    body {
      margin: 0;
      min-height: 100vh;
      background-color: #E0F7FA;
      background-image:
        url("data:image/svg+xml,%3Csvg width='180' height='180' viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cpath d='M35 55c-6.6 0-12-5.4-12-12 0-5.8 4.1-10.6 9.6-11.8C34.5 24.8 40.4 20 47.5 20c8.6 0 15.5 6.9 15.5 15.5 0 .6 0 1.2-.1 1.8 4.6 1.5 8.1 5.8 8.1 11 0 6.5-5.2 11.7-11.7 11.7H35z' fill='%23FFFFFF' fill-opacity='0.85'/%3E%3Cpath d='M135 32l3.2 6.5 7.2 1-5.2 5.1 1.2 7.2-6.4-3.4-6.4 3.4 1.2-7.2-5.2-5.1 7.2-1z' fill='%23FDE047' fill-opacity='0.9'/%3E%3Cpath d='M45 130c-3.5-4.5-10.5-3.5-12.5 1.5-2-5-9-6-12.5-1.5-3.5 4.5-1 11 12.5 18 13.5-7 16-13.5 12.5-18z' fill='%23F9A8D4' fill-opacity='0.75'/%3E%3Cpath d='M130 125q5 0 5-5 0 5 5 5-5 0-5 5 0-5-5-5z' stroke='%235EEAD4' stroke-width='3' stroke-linecap='round'/%3E%3Ccircle cx='90' cy='85' r='4' fill='%23FDBA74' fill-opacity='0.7'/%3E%3Ccircle cx='160' cy='95' r='3.5' fill='%2393C5FD' fill-opacity='0.8'/%3E%3Ccircle cx='18' cy='90' r='3.5' fill='%23C4B5FD' fill-opacity='0.7'/%3E%3C/g%3E%3C/svg%3E"),
        linear-gradient(135deg, #E0F2FE 0%, #FEF9C3 50%, #FCE7F3 100%);
      font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
      color: #334155;
      padding: 16px 10px 32px;
      display: flex;
      justify-content: center;
      align-items: flex-start;
    }

    .app-shell {
      max-width: 440px;
      width: 95%;
      background-color: #FFFDF7;
      background-image: radial-gradient(#FDE68A 1.2px, transparent 1.2px), radial-gradient(#BAE6FD 1.2px, #FFFDF7 1.2px);
      background-size: 36px 36px;
      background-position: 0 0, 18px 18px;
      padding: 1.5rem 1.25rem 1.8rem;
      border-radius: 36px;
      border: 7px solid #FFFFFF;
      box-shadow: 0 18px 40px rgba(56, 189, 248, 0.22), 0 0 0 3px #BAE6FD;
      animation: popIn 0.35s ease;
    }

    h1, h2, h3 {
      font-family: 'Arial Rounded MT Bold', 'Quicksand', 'Segoe UI', sans-serif;
      font-weight: 800;
      color: #0369A1;
      letter-spacing: 0.3px;
      margin-top: 0;
    }

    .step-indicator {
      text-align: center;
      font-size: 0.82rem;
      font-weight: 800;
      color: #9A3412;
      background: linear-gradient(90deg, #FEF08A 0%, #FED7AA 100%);
      padding: 0.35rem 1rem;
      border-radius: 999px;
      border: 2px solid #FDE047;
      display: inline-block;
      margin-bottom: 0.45rem;
      box-shadow: 0 3px 8px rgba(250, 204, 21, 0.25);
    }

    .offline-badge {
      display: inline-block;
      background-color: #DCFCE7;
      color: #166534;
      border: 1.5px solid #86EFAC;
      font-size: 0.72rem;
      font-weight: 800;
      padding: 0.2rem 0.65rem;
      border-radius: 999px;
      margin-bottom: 0.4rem;
    }

    .mobile-card {
      background-color: rgba(255, 255, 255, 0.96);
      border-radius: 24px;
      padding: 1.1rem;
      margin-bottom: 0.85rem;
      border: 2.5px solid #F1F5F9;
      box-shadow: 0 6px 14px rgba(148, 163, 184, 0.1);
    }
    .card-blue { background-color: #F0F9FF; border-color: #7DD3FC; }
    .card-peach { background-color: #FFF7ED; border-color: #FDBA74; }
    .card-mint { background-color: #ECFDF5; border-color: #6EE7B7; }
    .card-pink { background-color: #FFF1F2; border-color: #FDA4AF; }
    .card-yellow { background-color: #FEFCE8; border-color: #FDE047; }

    /* Form Controls */
    label.field-lbl { display: block; font-size: 0.83rem; font-weight: 700; color: #334155; margin: 0.55rem 0 0.25rem; }
    input[type="text"], input[type="number"], select {
      width: 100%; padding: 0.65rem 0.8rem; border-radius: 14px; border: 2px solid #CBD5E1;
      font-size: 0.92rem; background: #FFFFFF; color: #1E293B; outline: none;
    }
    input:focus, select:focus { border-color: #38BDF8; box-shadow: 0 0 0 3px rgba(56,189,248,0.2); }
    .row-2 { display: grid; grid-template-columns: 1.4fr 1fr; gap: 0.6rem; }

    /* Tabs */
    .tabs { display: flex; gap: 0.5rem; margin: 0.4rem 0 0.7rem; background: #E0F2FE; padding: 0.3rem; border-radius: 16px; }
    .tab-btn {
      flex: 1; padding: 0.55rem; border: none; border-radius: 12px; font-weight: 700; font-size: 0.85rem;
      background: transparent; color: #0369A1; cursor: pointer; transition: 0.2s;
    }
    .tab-btn.active
