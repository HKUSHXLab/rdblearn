# Installing use-rdblearn for Code Agents

Current RBDLearn version: 0.1.0
Make sure you are working with the right version

## Prerequisites

- [OpenCode.ai](https://opencode.ai), or Codex, Claude Code, or Cursor installed
- Git installed

## Installation Steps

### 1. Create the skill folder
Create a folder so agents' native skill tool discovers use-rdblearn skill, the `<skill-dir>` for Codex is `~/.codex/`, and for `claude code` is `~/.claude/`, and for `opencode` is `~/.opencode`.

```bash
rm -rf ~/<skill-dir>/skills/use-rdblearn
mkdir ~/<skill-dir>/skills/use-rdblearn
mkdir ~/<skill-dir>/skills/use-rdblearn/codes

```

### 2. Clone RDBLearn and FastDFS

```bash
git clone https://github.com/HKUSHXLab/rdblearn.git ~/<skill-dir>/skills/use-rdblearn/codes/rdblearn/
git clone https://github.com/HKUSHXLab/fastdfs.git ~/<skill-dir>/skills/use-rdblearn/codes/fastdfs/
```

### 2. Create the Skill

Create `~/<skill-dir>/skills/use-rdblearn/SKILL.md`:

```markdown
---
name: use-rdblearn
description: developement based on rdblearn
---

# Use RDBLearn
When asked about running rdblearn, always following:

1. Read the README and examples from `use-rdblearn/docs`.
2. Install RDBLearn
3. Write python code w.r.t. user requests
4. Run the code and debug if error raised
```

Move related documents to the skill directory

```bash
cp -r ~/<skill-dir>/skills/use-rdblearn/codes/rdblearn/README.md -r ~/<skill-dir>/skills/use-rdblearn/docs/rdblearn_README.md
cp -r ~/<skill-dir>/skills/use-rdblearn/codes/rdblearn/examples -r ~/<skill-dir>/skills/use-rdblearn/docs/rdblearn_examples
cp -r ~/<skill-dir>/skills/use-rdblearn/codes/fastdfs/README.md -r ~/<skill-dir>/skills/use-rdblearn/docs/fastdfs_README.md
cp -r ~/<skill-dir>/skills/use-rdblearn/codes/fastdfs/examples -r ~/<skill-dir>/skills/use-rdblearn/docs/fastdfs_examples
```

## Troubleshooting

### Skills not found

1. Makre sure the skill exists: `~/<skill-dir>/skills/use-rdblearn` and there is a `SKILL.md`
2. Use `skill` tool to list what's discovered


## Getting Help

- Report issues: https://github.com/HKUSHXLab/rdblearn/issues
- Full documentation: https://github.com/HKUSHXLab/rdblearn/main/skills/INSTALL.md