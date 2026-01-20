import { createBrowserRouter } from "react-router-dom";
import { AppShell } from "./App";
import { QuestionListPage } from "../pages/QuestionListPage";
import { PracticePage } from "../pages/PracticePage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <QuestionListPage /> },
      { path: "practice/:id", element: <PracticePage /> }
    ]
  }
]);