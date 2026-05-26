# Changesets

This folder is managed by [Changesets](https://github.com/changesets/changesets).

When you make a user-visible change to the package, run:

```bash
npx changeset
```

…and follow the prompts. A markdown file in this directory will record the change. The release workflow merges these into `CHANGELOG.md` and bumps the version when a release PR is merged to `main`.
