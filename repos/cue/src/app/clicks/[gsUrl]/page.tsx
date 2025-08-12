import { HydrateClient } from "~/trpc/server";
import { ClicksDataPageContent } from "./clicks-data-page-content";

interface ClicksDataPageProps {
  params: Promise<{ gsUrl: string }>;
}

export default async function ClicksDataPage({ params }: ClicksDataPageProps) {
  const { gsUrl } = await params;
  const decodedPath = decodeURIComponent(gsUrl);

  return (
    <HydrateClient>
      <ClicksDataPageContent gsUrl={gsUrl} decodedPath={decodedPath} />
    </HydrateClient>
  );
}