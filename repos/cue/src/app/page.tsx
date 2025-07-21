import { api, HydrateClient } from "~/trpc/server";
import { QueueViewer } from "~/components/queue-viewer";

export default async function Home() {
  // void api.post.getLatest.prefetch();
  void api.kueue.getJobs.prefetch();

  return (
    <HydrateClient>
      <QueueViewer />
    </HydrateClient>
  );
}
