import { api } from "~/trpc/server";
import { ClicksContextProvider } from "./clicks-context";

interface ClicksLayoutProps {
  children: React.ReactNode;
  params: Promise<{ gsUrl: string }>;
}

export default async function ClicksLayout({ children, params }: ClicksLayoutProps) {
  const { gsUrl } = await params;
  const decodedPath = decodeURIComponent(gsUrl);

  let clicksData;
  let error;

  try {
    clicksData = await api.clicks.getClickEvalData({ filePath: decodedPath });
  } catch (e) {
    error = e instanceof Error ? e.message : "Failed to load evaluation data";
  }

  return (
    <ClicksContextProvider data={clicksData} error={error}>
      {children}
    </ClicksContextProvider>
  );
}