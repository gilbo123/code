
<template>
  <div id="app">
    <h1>Strawberry Dashboard</h1>
    <nav>
        <router-link to="/"> Home </router-link>
        <router-link to="/bar"> | Bar Chart </router-link>
        <router-link to="/pie">  | Pie Chart </router-link>
    </nav>
    <!-- route outlet -->
    <!-- component matched by the route will render here -->
    <router-view></router-view>
  </div>
</template>

<script>
export default {
}
</script>


<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
