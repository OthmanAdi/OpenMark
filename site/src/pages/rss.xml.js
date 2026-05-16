import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';

export async function GET(context) {
  const issues = await getCollection('newsletters', ({ data }) => !data.draft);
  return rss({
    title: 'OpenMark',
    description: 'Curated weekly newsletter from a 13k-bookmark knowledge graph.',
    site: context.site,
    items: issues
      .sort((a, b) => b.data.published_at.getTime() - a.data.published_at.getTime())
      .map((issue) => ({
        title: issue.data.title,
        description: issue.data.subtitle || '',
        pubDate: issue.data.published_at,
        link: `/issues/${issue.slug}/`,
      })),
    customData: '<language>en-us</language>',
  });
}
