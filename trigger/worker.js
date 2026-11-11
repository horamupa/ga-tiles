// Starts the tile builds every hour. GitHub's own hourly schedule starts only 3 to 5 runs a day, so the site lagged GFS
// by up to 13 hours (B-118). The cron of wrangler.toml asks GitHub for a workflow_dispatch run instead; a run that finds
// nothing new ends in about a minute. Add a workflow here once it is on main.
//
// Deploy from this folder (wrangler asks to log in to Cloudflare the first time):
//   npx --yes wrangler@4 deploy
//   npx --yes wrangler@4 secret put GITHUB_TOKEN   # fine-grained token: repository ga-tiles, Actions read and write
const WORKFLOWS = ["tiles.yml"];

export default {
  async scheduled(controller, env) {
    const failed = [];
    for (const workflow of WORKFLOWS) {
      const response = await fetch(
        `https://api.github.com/repos/horamupa/ga-tiles/actions/workflows/${workflow}/dispatches`,
        {
          method: "POST",
          headers: {
            Accept: "application/vnd.github+json",
            Authorization: `Bearer ${env.GITHUB_TOKEN}`,
            "User-Agent": "ga-tiles-trigger",
            "X-GitHub-Api-Version": "2022-11-28",
          },
          body: JSON.stringify({ ref: "main" }),
        },
      );
      if (!response.ok) failed.push(`${workflow}: GitHub answered ${response.status} ${await response.text()}`);
    }
    // The cron event shows as failed in the Cloudflare dashboard; .github/workflows/fresh.yml mails the owner.
    if (failed.length > 0) throw new Error(failed.join("\n"));
  },
};
