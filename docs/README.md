# Documentation

Welcome to the **Multi-Agent Task Allocation & Adaptation** project documentation!

---

## 📚 Documentation Structure

### 🚀 Getting Started
**Start here if you're new to the project**

- [**Quick Start Guide**](getting-started/QUICK_START_GUIDE.md) - Get up and running in 5 minutes

### 📖 API Reference
**Detailed API documentation for all components**

- [**MATCH-AOU API**](api-reference/MATCH_AOU_API.md) - Task allocation solver
- [**BLADE API**](api-reference/BLADE_API_DOCUMENTATION.md) - Simulation engine
- [**Integration Guide**](api-reference/INTEGRATION_GUIDE.md) - Connecting MATCH-AOU with BLADE

### 🗂️ Project Information
**Understanding the codebase structure**

- [**Project Structure**](project-info/PROJECT_STRUCTURE.md) - File organization and module overview
- [**Imports & Dependencies**](project-info/IMPORTS_AND_DEPENDENCIES.md) - Module relationships

---

## 🎯 Quick Navigation

### I want to...

**...understand the project**  
→ Start with [Quick Start Guide](getting-started/QUICK_START_GUIDE.md)

**...use MATCH-AOU solver**  
→ See [MATCH-AOU API](api-reference/MATCH_AOU_API.md)

**...use BLADE simulation**  
→ See [BLADE API](api-reference/BLADE_API_DOCUMENTATION.md)

**...connect MATCH-AOU and BLADE**  
→ See [Integration Guide](api-reference/INTEGRATION_GUIDE.md)

**...find a specific file**  
→ See [Project Structure](project-info/PROJECT_STRUCTURE.md)

**...understand module dependencies**  
→ See [Imports & Dependencies](project-info/IMPORTS_AND_DEPENDENCIES.md)

---

## 🔍 Documentation by Component

### MATCH-AOU (Solver)
- **API:** [MATCH_AOU_API.md](api-reference/MATCH_AOU_API.md)
- **Models:** Agent, Task, Step, StepType, Capability, Location
- **Solver:** MatchAou class, MINLP optimization

### BLADE (Simulation)
- **API:** [BLADE_API_DOCUMENTATION.md](api-reference/BLADE_API_DOCUMENTATION.md)
- **Core:** Game, Scenario, Side
- **Units:** Aircraft, Ship, Facility, Airbase, Weapon
- **Missions:** StrikeMission, PatrolMission

### Integration (blade_utils)
- **Guide:** [INTEGRATION_GUIDE.md](api-reference/INTEGRATION_GUIDE.md)
- **Utils:** blade_plan_utils, blade_executor, scenario_factory, observation_utils

---

## 📊 Documentation Statistics

- **Total Documentation:** ~80KB
- **API References:** 3 (MATCH-AOU, BLADE, Integration)
- **Getting Started Guides:** 1
- **Project Info Docs:** 2
- **Total Files:** 49 Python files, 8 packages

---

## 🔄 Keeping Documentation Updated

### When to Update

| Documentation | When to Update |
|---------------|----------------|
| **Quick Start Guide** | When workflow changes |
| **MATCH-AOU API** | When solver interface changes |
| **BLADE API** | When BLADE methods change |
| **Integration Guide** | When blade_utils changes |
| **Project Structure** | When adding/removing files |
| **Imports & Dependencies** | When changing module structure |

### How to Update

Most documentation includes examples based on current code. Update examples when APIs change.

---

## 💡 Tips for Using This Documentation

1. **Start with Quick Start** - Don't dive into API docs first
2. **Use cross-references** - Documents link to each other
3. **Search by topic** - Use Ctrl+F within documents
4. **Keep it handy** - Reference while coding
5. **Read code examples** - They're tested and practical

---

## 🤝 Contributing

When adding new features:
1. Update relevant API documentation
2. Add examples to demonstrate usage
3. Update Integration Guide if it affects MATCH-AOU ↔ BLADE
4. Keep cross-references up to date

---

## 📞 Need Help?

- **Can't find something?** Check the [Quick Navigation](#-quick-navigation) section above
- **API unclear?** Each API doc has a "Complete Example" section
- **Integration issues?** See [Integration Guide](api-reference/INTEGRATION_GUIDE.md) troubleshooting section

---

**Last Updated:** February 2026  
**Project Status:** MATCH-AOU and BLADE integrated, RL layer in development  
**Documentation Version:** 2.0 (Organized Structure)
