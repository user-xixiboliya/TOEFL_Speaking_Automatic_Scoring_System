import { Outlet } from "react-router-dom";
import { TopBar } from "../components/TopBar";

export function AppShell() {
  return (
    <div className="min-h-full bg-slate-50">
      <TopBar />
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}